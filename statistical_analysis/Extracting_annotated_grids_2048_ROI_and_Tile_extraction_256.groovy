import qupath.lib.scripting.QP
import qupath.lib.regions.RegionRequest
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.projects.ProjectImageEntry
import qupath.lib.roi.interfaces.ROI

import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption

// =========================
// USER PARAMETERS
// =========================
String OUTPUT_ROOT = "E:/WSI_ATTN_ANALYSIS/Selected_2048_patch"  // <-- CHANGE
double DOWNSAMPLE = 1.0

int GRID_SIZE  = 1024     // grid cell size
int PATCH_SIZE = 1024     // patch size
int TILE_SIZE  = 512      // tile size, determine based on MPP
int TILES_PER_SIDE = (int)(PATCH_SIZE / TILE_SIZE)   // 2

boolean DEDUP_GRID_CELLS = true  // if two ellipses fall in same grid cell, keep only one

// =========================
// HELPERS
// =========================
def safeName(String s) { s.replaceAll("[\\\\/:*?\"<>|]", "_") }
def ensureDir(def p) { Files.createDirectories(p) }

def isEllipseMarker(ROI roi) {
    // robust across QuPath versions
    return roi.getClass().getSimpleName().toLowerCase().contains("ellipse")
}

def compareCentroidTopLeft(PathAnnotationObject a, PathAnnotationObject b) {
    def ra = a.getROI()
    def rb = b.getROI()
    int cy = (ra.getCentroidY() <=> rb.getCentroidY())
    return (cy != 0) ? cy : (ra.getCentroidX() <=> rb.getCentroidX())
}

// =========================
// MAIN: iterate project images
// =========================
def project = QP.getProject()
if (project == null) {
    println "❌ No project open."
    return
}

ensureDir(Paths.get(OUTPUT_ROOT))

def imageList = project.getImageList()
println "Found ${imageList.size()} images in project."

imageList.eachWithIndex { ProjectImageEntry entry, int imgIdx ->

    def imageData = entry.readImageData()
    if (imageData == null) {
        println "⚠️ Could not read image data for entry: ${entry}"
        return
    }

    def server = imageData.getServer()
    def hierarchy = imageData.getHierarchy()

    def imageNameRaw = entry.getImageName()
    def imageName = safeName(imageNameRaw.replaceAll('\\.(svs|tif|tiff|ndpi|scn|mrxs)$', ''))

    println "\n[${imgIdx+1}/${imageList.size()}] Processing: ${imageName}"

    def width  = server.getWidth()
    def height = server.getHeight()

    // Microns per pixel (for filename microns field)
    def cal = server.getPixelCalibration()
    double umPerPx = (cal != null && cal.hasPixelSizeMicrons()) ? cal.getPixelWidthMicrons() : Double.NaN

    // Output structure you requested:
    // OUTPUT_ROOT / imageName / patches   (ALL TIFFs)
    // OUTPUT_ROOT / imageName / tiles / patchXX  (tiles for each patch)
    def wsiRoot   = Paths.get(OUTPUT_ROOT, imageName)
    def patchesDir = wsiRoot.resolve("patches")
    def tilesRoot  = wsiRoot.resolve("tiles")
    ensureDir(patchesDir)
    ensureDir(tilesRoot)

    // CSV index per WSI
    def csvPath = wsiRoot.resolve("${imageName}_index.csv")
    def header = "slide_name,patch_id,gridX,gridY,patch_x,patch_y,patch_w,patch_h,tile_row,tile_col,tile_x,tile_y,tile_w,tile_h\n"
    Files.write(csvPath, header.getBytes("UTF-8"))

    // ---- Find ellipse markers
    def annots = hierarchy.getAnnotationObjects().findAll { it instanceof PathAnnotationObject } as List<PathAnnotationObject>
    def markers = annots.findAll { a -> isEllipseMarker(a.getROI()) }

    if (markers.isEmpty()) {
        println "⚠️ No ellipse annotations found -> SKIP ${imageName}"
        return
    }

    // sort markers top-left
    markers.sort { a, b -> compareCentroidTopLeft(a, b) }

    // Convert markers -> grid cells
    def gridCells = []
    for (m in markers) {
        def roi = m.getROI()
        int gridX = (int)Math.floor(roi.getCentroidX() / GRID_SIZE)
        int gridY = (int)Math.floor(roi.getCentroidY() / GRID_SIZE)
        gridCells << [gridX, gridY]
    }

    // Optional de-dup (avoid exporting same grid cell twice)
    if (DEDUP_GRID_CELLS) {
        def seen = new HashSet<String>()
        def unique = []
        for (cell in gridCells) {
            def key = "${cell[0]}_${cell[1]}"
            if (!seen.contains(key)) {
                seen.add(key)
                unique << cell
            }
        }
        gridCells = unique
    }

    println "✅ Found ${gridCells.size()} unique grid cells with ellipse markers"
    println "✅ Using grid cells: " + gridCells.collect{ "[${it[0]},${it[1]}]" }.join(", ")

    // ---- Export ALL selected cells
    gridCells.eachWithIndex { cell, int pIdx ->

        int gridX = cell[0]
        int gridY = cell[1]

        int patchX = gridX * GRID_SIZE
        int patchY = gridY * GRID_SIZE

        // bounds check
        if (patchX < 0 || patchY < 0 || patchX + PATCH_SIZE > width || patchY + PATCH_SIZE > height) {
            println "⚠️ Patch outside bounds for cell [${gridX},${gridY}] -> SKIP patch${pIdx+1}"
            return
        }

        // Create full patch name FIRST
        double patchMicronSize = Double.isNaN(umPerPx) ? 0.0 : (PATCH_SIZE * DOWNSAMPLE * umPerPx)

        def patchId = String.format(
                "%s_patch%02d_x%d_y%d_tileX%d_tileY%d_%dx%d_microns",
                imageName, pIdx + 1,
                patchX, patchY,
                gridX, gridY,
                (int)Math.round(patchMicronSize), (int)Math.round(patchMicronSize)
        )

        // Tile folder: tiles/test_064_patch01_x24576_y114688_tileX12_tileY56_498x498_microns/
        def tilesDir = tilesRoot.resolve(patchId)
        ensureDir(tilesDir)
        
        // ---- Export TIFF patch into patches folder
        BufferedImage patchImg = null
        try {
            def patchReq = RegionRequest.createInstance(imageData.getServerPath(), DOWNSAMPLE, patchX, patchY, PATCH_SIZE, PATCH_SIZE)
            patchImg = server.readRegion(patchReq)
        
            def patchFileName = "${patchId}.tif"

            def patchPath = patchesDir.resolve(patchFileName)
            ImageIO.write(patchImg, "TIFF", patchPath.toFile())
            println "✅ Patch ${pIdx+1}/${gridCells.size()} TIFF: ${patchPath}"

        } catch (Exception e) {
            println "❌ Patch export error ${imageName} ${patchId}: ${e.getMessage()}"
            e.printStackTrace()
            return
        } finally {
            if (patchImg != null) patchImg.flush()
        }

        // ---- Export PNG tiles into tiles/patchXX/
        for (int r = 0; r < TILES_PER_SIDE; r++) {
            for (int c = 0; c < TILES_PER_SIDE; c++) {

                BufferedImage tileImg = null
                try {
                    int pixelX = patchX + c * TILE_SIZE
                    int pixelY = patchY + r * TILE_SIZE

                    def tileReq = RegionRequest.createInstance(imageData.getServerPath(), DOWNSAMPLE, pixelX, pixelY, TILE_SIZE, TILE_SIZE)
                    tileImg = server.readRegion(tileReq)

                    double tileMicronSize = Double.isNaN(umPerPx) ? 0.0 : (TILE_SIZE * DOWNSAMPLE * umPerPx)

                    def tileFileName = String.format(
                            "%s_%s_x%d_y%d_tileX%d_tileY%d_%dx%d_microns.png",
                            imageName, patchId,
                            pixelX, pixelY,     // raw pixel coords
                            c, r,               // tile indices in the patch
                            (int)Math.round(tileMicronSize), (int)Math.round(tileMicronSize)
                    )

                    def tilePath = tilesDir.resolve(tileFileName)
                    ImageIO.write(tileImg, "PNG", tilePath.toFile())

                    def line = "${imageName},${patchId},${gridX},${gridY},${patchX},${patchY},${PATCH_SIZE},${PATCH_SIZE},${r},${c},${pixelX},${pixelY},${TILE_SIZE},${TILE_SIZE}\n"
                    Files.write(csvPath, line.getBytes("UTF-8"), StandardOpenOption.APPEND)

                } catch (Exception e) {
                    println "❌ Tile export error ${imageName} ${patchId} r=${r} c=${c}: ${e.getMessage()}"
                } finally {
                    if (tileImg != null) tileImg.flush()
                }
            }
        }
    }

    println "🎉 Completed: ${imageName} - Exported ${gridCells.size()} patches"
}

println "\nDONE. Output root: ${OUTPUT_ROOT}"