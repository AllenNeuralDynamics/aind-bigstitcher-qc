package org.aind.bigstitcher.qc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import mpicbg.spim.data.generic.sequence.BasicViewDescription;
import mpicbg.spim.data.sequence.ViewId;
import mpicbg.spim.data.sequence.ViewSetup;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.basictypeaccess.array.ByteArray;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import net.preibisch.mvrecon.Threads;
import net.preibisch.mvrecon.fiji.spimdata.SpimData2;
import net.preibisch.mvrecon.process.downsampling.DownsampleTools;
import net.preibisch.mvrecon.process.fusion.FusionTools;
import net.preibisch.mvrecon.process.fusion.lazy.LazyFusionTools;
import net.preibisch.mvrecon.process.fusion.transformed.TransformView;
import net.preibisch.mvrecon.process.n5api.N5ApiTools;
import net.preibisch.mvrecon.process.n5api.N5ApiTools.MultiResolutionLevelInfo;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.universe.StorageFormat;
import util.URITools;

final class CompositeUtils {

    private CompositeUtils() {
    }

    enum ColorMode {
        VIEW,
        ILLUMINATION
    }

    static ColorMode parseColorMode(final String value) {
        if (value == null)
            return ColorMode.ILLUMINATION;
        final String normalized = value.trim().toLowerCase();
        switch (normalized) {
            case "view":
                return ColorMode.VIEW;
            case "illumination":
                return ColorMode.ILLUMINATION;
            default:
                throw new IllegalArgumentException("Unsupported color mode: " + value);
        }
    }

    static CompositeBlockRenderer createCompositeBlockRenderer(
            final SpimData2 spimData,
            final Map<ViewId, AffineTransform3D> regs,
            final Collection<? extends ViewId> allViews,
            final Interval cropBB,
            final int interpolation,
            final ColorMode colorMode,
            final float displayMax
    ) {
        final Map<ViewId, ? extends BasicViewDescription<ViewSetup>> vds =
                spimData.getSequenceDescription().getViewDescriptions();

        final ArrayList<ViewId> overlapping = LazyFusionTools.overlappingViewIds(
                cropBB,
                allViews,
                regs,
                LazyFusionTools.assembleDimensions(allViews, vds),
                LazyFusionTools.defaultAffineExpansion
        );

        final List<RandomAccessibleInterval<FloatType>> transformedSources = new ArrayList<>(overlapping.size());
        final List<Integer> sourceIds = new ArrayList<>(overlapping.size());
        final List<long[]> sourceBounds = new ArrayList<>(overlapping.size()); // [minX,minY,minZ,maxX,maxY,maxZ] in zero-min overlay coords
        final ColorMode effectiveMode = colorMode == null ? ColorMode.ILLUMINATION : colorMode;

        for (final ViewId vid : overlapping) {
            final AffineTransform3D model = regs.get(vid).copy();

            final double[] usedDownsampleFactors = new double[3];
            final RandomAccessibleInterval<FloatType> input = DownsampleTools.openDownsampled(
                    spimData.getSequenceDescription().getImgLoader(),
                    vid,
                    model,
                    usedDownsampleFactors
            );

            RandomAccessibleInterval<FloatType> transformed = TransformView.transformView(
                    input,
                    model,
                    cropBB,
                    0,
                    interpolation
            );

            transformed = Views.zeroMin(transformed);
            transformedSources.add(transformed);

            // Compute this source's bounds within the overlay (zero-min) coordinates.
            sourceBounds.add(Utils.computeSourceBounds(input, model, cropBB));

            final BasicViewDescription<ViewSetup> vd = vds.get(vid);
            final int sourceId = effectiveMode == ColorMode.ILLUMINATION
                    ? Utils.illuminationIdOf(vd)
                    : vid.getViewSetupId();
            sourceIds.add(sourceId);
        }

        final int n = transformedSources.size();
        final int[] palette;
        final int[] colorIndexForSource = new int[n];

        if (effectiveMode == ColorMode.ILLUMINATION) {
            final LinkedHashMap<Integer, Integer> idToIndex = new LinkedHashMap<>();
            for (final Integer id : sourceIds)
                idToIndex.computeIfAbsent(id, ignored -> idToIndex.size());
            palette = distinctPalette(idToIndex.size());
            for (int idx = 0; idx < n; idx++)
                colorIndexForSource[idx] = idToIndex.get(sourceIds.get(idx));
        } else {
            palette = distinctPalette(n);
            for (int idx = 0; idx < n; idx++)
                colorIndexForSource[idx] = idx;
        }

        final Interval outInterval = FusionTools.getFusedZeroMinInterval(cropBB);
        final long[] cropMin = new long[3];
        cropBB.min(cropMin);

        final long[][] boundsArray = sourceBounds.toArray(new long[0][]);
        return new CompositeBlockRenderer(outInterval, transformedSources, palette, colorIndexForSource, cropMin, displayMax, boundsArray);
    }

    static Path exportCompositeAsOmeZarr(
            final CompositeBlockRenderer overlay,
            final Path outputRoot,
            final String relativeName,
            final Integer userDefinedBlockSize,
            final int maxDownsamplingLevels,
            final String unit
    ) throws IOException {
        if (outputRoot == null)
            throw new IllegalStateException("OME-Zarr output root is not configured");

        final Path outputFolder = outputRoot.resolve(relativeName).normalize();
        final Path containerPath = outputFolder.getParent() == null
                ? Paths.get(outputFolder.toString() + ".ome.zarr")
                : outputFolder.getParent().resolve(outputFolder.getFileName().toString() + ".ome.zarr");

        Files.createDirectories(containerPath.getParent());
        Utils.deleteRecursivelyIfExists(containerPath);

        final long[] spatialDims = overlay.spatialDimensions();
        final long[] dims5d = new long[] { spatialDims[0], spatialDims[1], spatialDims[2], 3, 1 };
        final int[] blockSize = ZarrUtils.defaultBlockSize(spatialDims, userDefinedBlockSize);
        final int[][] downsamplings = ZarrUtils.defaultDownsamplings(spatialDims, maxDownsamplingLevels);
        final Compression compression = new GzipCompression(5);

        final ExecutorService executor = Executors.newFixedThreadPool(Threads.numThreads());
        try {
            try (final N5Writer writer = URITools.instantiateN5Writer(
                    StorageFormat.ZARR,
                    URITools.toURI(containerPath.toString()))) {

                final MultiResolutionLevelInfo[] pyramid = N5ApiTools.setupMultiResolutionPyramid(
                        writer,
                        level -> Integer.toString(level),
                        DataType.UINT8,
                        dims5d,
                        compression,
                        blockSize,
                        downsamplings
                );

                ZarrUtils.writeS0Blocks(overlay, writer, pyramid[0], executor);
                ZarrUtils.writeDownsampledLevels(writer, pyramid, executor);
                ZarrUtils.writeOmeNgffMetadata(
                        writer,
                        pyramid,
                        containerPath.getFileName().toString(),
                        overlay.worldMin3d(),
                        unit
                );
            }
        } finally {
            ZarrUtils.shutdownExecutor(executor);
        }

        return containerPath;
    }

    interface BlockRenderer {
        long[] spatialDimensions();

        void renderBlock(long[] blockMin, ArrayImg<UnsignedByteType, ByteArray> target);
    }

    static final class CompositeBlockRenderer implements BlockRenderer {
        private final List<RandomAccessibleInterval<FloatType>> sources;
        private final int[] palette;
        private final int[] colorIndexForSource;
        private final long[] spatialDims;
        private final double[] worldMin3d;
        private final float displayMax;
        private final long[][] sourceBounds; // [minX,minY,minZ,maxX,maxY,maxZ] per source in zero-min overlay coords

        CompositeBlockRenderer(
                final Interval interval,
                final List<RandomAccessibleInterval<FloatType>> sources,
                final int[] palette,
                final int[] colorIndexForSource,
                final long[] cropMin,
                final float displayMax,
                final long[][] sourceBounds
        ) {
            this.sources = new ArrayList<>(sources);
            this.palette = palette;
            this.colorIndexForSource = colorIndexForSource.clone();
            this.spatialDims = new long[3];
            interval.dimensions(this.spatialDims);
            this.worldMin3d = new double[] {
                    cropMin != null && cropMin.length > 0 ? cropMin[0] : 0.0,
                    cropMin != null && cropMin.length > 1 ? cropMin[1] : 0.0,
                    cropMin != null && cropMin.length > 2 ? cropMin[2] : 0.0
            };
            this.displayMax = displayMax;
            this.sourceBounds = sourceBounds;
        }

        @Override
        public long[] spatialDimensions() {
            return spatialDims.clone();
        }

        double[] worldMin3d() {
            return worldMin3d.clone();
        }

        @Override
        public void renderBlock(final long[] blockMin, final ArrayImg<UnsignedByteType, ByteArray> target) {
            final int sizeX = (int) target.dimension(0);
            final int sizeY = (int) target.dimension(1);
            final int sizeZ = (int) target.dimension(2);
            if (sizeX <= 0 || sizeY <= 0 || sizeZ <= 0)
                return;

            final ByteArray access = (ByteArray) target.update(null);
            final byte[] data = access.getCurrentStorageArray();
            final int sliceSize = sizeX * sizeY;
            final int volumeSize = sliceSize * sizeZ;
            final int[] channelOffsets = new int[] { 0, volumeSize, 2 * volumeSize };

            final int sourceCount = sources.size();
            final List<RandomAccessibleInterval<FloatType>> localSources = this.sources;
            final int[] paletteLocal = this.palette;
            final int[] colorIndex = this.colorIndexForSource;
            final float localDisplayMax = this.displayMax;

            // Determine active sources for this block by intersecting per-source bounds with the block bounds.
            final List<Integer> activeSources = Utils.collectActiveSourcesForBlock(
                    blockMin,
                    sizeX,
                    sizeY,
                    sizeZ,
                    sourceBounds,
                    sourceCount
            );

            if (activeSources.isEmpty())
                return;

            final int activeCountFinal = activeSources.size();
            // Convert to primitive indices once to avoid autoboxing in the inner loops.
            final int[] activeIdx = new int[activeCountFinal];
            for (int i = 0; i < activeCountFinal; i++)
                activeIdx[i] = activeSources.get(i);

            final ArrayList<RandomAccess<FloatType>> ras = new ArrayList<>(activeCountFinal);
            for (int i = 0; i < activeCountFinal; i++)
                ras.add(localSources.get(activeIdx[i]).randomAccess());

            final long[] pos = new long[] { 0L, 0L, 0L };

            for (int localZ = 0; localZ < sizeZ; localZ++) {
                pos[2] = blockMin[2] + localZ;
                final int sliceOffset = localZ * sizeY;

                for (int localY = 0; localY < sizeY; localY++) {
                    pos[1] = blockMin[1] + localY;
                    final int rowOffset = (sliceOffset + localY) * sizeX;

                    for (int localX = 0; localX < sizeX; localX++) {
                        pos[0] = blockMin[0] + localX;
                        int argb = 0;
                        for (int i = 0; i < activeCountFinal; i++) {
                            final RandomAccess<FloatType> ra = ras.get(i);
                            ra.setPosition(pos);
                            final float val = ra.get().getRealFloat();
                            if (val <= 0)
                                continue;
                            final int level = clampTo8bit((val / localDisplayMax) * 255f);
                            if (level == 0)
                                continue;
                            final int srcColor = tint(paletteLocal[colorIndex[activeIdx[i]]], level);
                            argb = maxBlend(argb, srcColor);
                        }

                        if (argb == 0)
                            continue;

                        final int baseIndex = rowOffset + localX;
                        data[channelOffsets[0] + baseIndex] = (byte) ((argb >> 16) & 0xff);
                        data[channelOffsets[1] + baseIndex] = (byte) ((argb >> 8) & 0xff);
                        data[channelOffsets[2] + baseIndex] = (byte) (argb & 0xff);
                    }
                }
            }
        }
    }

    static int tint(final int baseColor, final int level) {
        final int br = (baseColor >> 16) & 0xff;
        final int bg = (baseColor >> 8)  & 0xff;
        final int bb = (baseColor)       & 0xff;
        final int r = (br * level) / 255;
        final int g = (bg * level) / 255;
        final int b = (bb * level) / 255;
        return (0xff << 24) | (r << 16) | (g << 8) | b;
    }

    static int maxBlend(final int dst, final int src) {
        final int r = Math.max((dst >> 16) & 0xff, (src >> 16) & 0xff);
        final int g = Math.max((dst >> 8)  & 0xff, (src >> 8)  & 0xff);
        final int b = Math.max(dst & 0xff,        src & 0xff);
        return (0xff << 24) | (r << 16) | (g << 8) | b;
    }

    static int clampTo8bit(final float v) {
        return (v <= 0) ? 0 : (v >= 255f ? 255 : (int) v);
    }

    static int[] distinctPalette(final int n) {
        final int[] base = new int[] {
                0xFF0000,
                0x00FF00,
                0x0000FF,
                0xFFFF00,
                0xFF00FF,
                0x00FFFF,
                0xFFA500,
                0x8000FF,
                0x00FF80,
                0xFF0080
        };
        final int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = base[i % base.length];
        return out;
    }
}
