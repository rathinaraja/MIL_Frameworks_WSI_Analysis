import qupath.lib.roi.RectangleROI
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.scripting.QP

// Parameters
int tileSize = 2048  // 2048 x 2048 pixels

// Get the project
def project = QP.getProject()
if (project == null) {
    print "No project open! Please open a project first."
    return
}

// Process each image in the project
project.getImageList().each { entry ->
    try {
        // Open image
        def imageData = entry.readImageData()
        def server = imageData.getServer()
        def hierarchy = imageData.getHierarchy()
        
        // Get image dimensions
        int width = server.getWidth()
        int height = server.getHeight()
        
        println "Processing: ${entry.getImageName()}"
        println "Image dimensions: ${width} x ${height}"
        
        // Create grid annotations
        for (int x = 0; x < width; x += tileSize) {
            for (int y = 0; y < height; y += tileSize) {
                def roi = new RectangleROI(x, y, 
                    Math.min(tileSize, width - x), 
                    Math.min(tileSize, height - y))
                def annotation = new PathAnnotationObject(roi)
                hierarchy.addObject(annotation)
            }
        }
        
        // Save the changes
        entry.saveImageData(imageData)
        println "✅ Grid added to ${entry.getImageName()}"
        
        // Clean up
        imageData.getServer().close()
        
    } catch (Exception e) {
        println "❌ Error processing ${entry.getImageName()}: ${e.getMessage()}"
        e.printStackTrace()
    }
}

// Save project
project.syncChanges()
println "🎉 Grid annotations added to all images in the project!"