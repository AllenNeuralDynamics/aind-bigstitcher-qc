package org.aind.bigstitcher.qc;

import bdv.ViewerImgLoader;
import bdv.cache.CacheControl;
import bdv.img.cache.VolatileGlobalCellCache;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import mpicbg.spim.data.generic.sequence.BasicViewDescription;
import mpicbg.spim.data.sequence.ImgLoader;
import mpicbg.spim.data.sequence.ViewDescription;
import mpicbg.spim.data.sequence.ViewId;
import mpicbg.spim.data.sequence.ViewSetup;
import net.imglib2.Dimensions;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.util.Intervals;
import net.imglib2.type.numeric.real.FloatType;
import net.preibisch.mvrecon.fiji.spimdata.SpimData2;

final class Utils {

    private Utils() {
    }

    static void deleteRecursivelyIfExists(final Path path) throws IOException {
        if (!Files.exists(path))
            return;

        try (Stream<Path> walk = Files.walk(path)) {
            walk.sorted(Comparator.reverseOrder()).forEach(p -> {
                try {
                    Files.deleteIfExists(p);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
    }

    static List<ViewId> collectPresentViews(final SpimData2 spimData) {
        final List<ViewId> views = new ArrayList<>();
        for (final ViewDescription vd : spimData.getSequenceDescription().getViewDescriptions().values()) {
            if (spimData.getSequenceDescription().getMissingViews() == null ||
                    !spimData.getSequenceDescription().getMissingViews().getMissingViews().contains(vd)) {
                views.add(vd);
            }
        }
        return views;
    }

    static Interval boundsForView(
            final ViewId vid,
            final Map<ViewId, AffineTransform3D> regs,
            final Map<ViewId, ? extends BasicViewDescription<?>> viewDescriptions
    ) {
        final BasicViewDescription<?> vd = viewDescriptions.get(vid);
        final AffineTransform3D model = regs.get(vid);
        if (vd == null || model == null)
            return null;

        final Dimensions size = vd.getViewSetup().getSize();
        if (size == null)
            return null;

        return Intervals.largestContainedInterval(model.estimateBounds(new FinalInterval(size)));
    }

    static int illuminationIdOf(final BasicViewDescription<ViewSetup> vd) {
        if (vd == null || vd.getViewSetup() == null || vd.getViewSetup().getIllumination() == null)
            return -1;
        return vd.getViewSetup().getIllumination().getId();
    }

    static void clearImgLoaderCache(final SpimData2 spimData) {
        final ImgLoader imgLoader = spimData.getSequenceDescription().getImgLoader();
        if (imgLoader instanceof ViewerImgLoader) {
            final CacheControl cacheControl = ((ViewerImgLoader) imgLoader).getCacheControl();
            if (cacheControl == null)
                return;
            if (cacheControl instanceof VolatileGlobalCellCache) {
                ((VolatileGlobalCellCache) cacheControl).clearCache();
                System.out.println("Clearing cache");
            } else {
                try {
                    cacheControl.getClass().getMethod("clearCache").invoke(cacheControl);
                    System.out.println("Clearing cache");
                } catch (Exception ignored) {
                    // ignore if cache cannot be cleared explicitly
                }
            }
        }
    }

    static List<Integer> collectActiveSourcesForBlock(
            final long[] blockMin,
            final int sizeX,
            final int sizeY,
            final int sizeZ,
            final long[][] sourceBounds,
            final int sourceCount
    ) {
        final long bx0 = blockMin[0];
        final long by0 = blockMin[1];
        final long bz0 = blockMin[2];
        final long bx1 = bx0 + sizeX - 1;
        final long by1 = by0 + sizeY - 1;
        final long bz1 = bz0 + sizeZ - 1;

        final List<Integer> active = new ArrayList<>();
        for (int s = 0; s < sourceCount; s++) {
            final long[] bounds = sourceBounds != null && s < sourceBounds.length ? sourceBounds[s] : null;
            if (bounds == null)
                continue;
            if (bounds[0] > bounds[3] || bounds[1] > bounds[4] || bounds[2] > bounds[5])
                continue;
            final boolean intersects = !(bounds[0] > bx1 || bounds[3] < bx0 ||
                                         bounds[1] > by1 || bounds[4] < by0 ||
                                         bounds[2] > bz1 || bounds[5] < bz0);
            if (intersects)
                active.add(s);
        }
        return active;
    }

    /* Determine the bounding box of the source in the zero-min coordinates */
    static long[] computeSourceBounds(
            final RandomAccessibleInterval<FloatType> input,
            final AffineTransform3D model,
            final Interval cropBB
    ) {
        try {
            final FinalInterval inputInterval = new FinalInterval(input);
            final Interval worldBounds = Intervals.largestContainedInterval(model.estimateBounds(inputInterval));
            final Interval clipped = Intervals.intersect(worldBounds, cropBB);
            final long[] bounds = new long[6];
            if (Intervals.isEmpty(clipped)) {
                bounds[0] = 1;
                bounds[1] = 1;
                bounds[2] = 1;
                bounds[3] = 0;
                bounds[4] = 0;
                bounds[5] = 0;
            } else {
                final long[] minWorld = new long[3];
                final long[] maxWorld = new long[3];
                clipped.min(minWorld);
                clipped.max(maxWorld);
                final long[] cropMinArr = new long[3];
                cropBB.min(cropMinArr);
                for (int d = 0; d < 3; d++) {
                    bounds[d] = Math.max(0, minWorld[d] - cropMinArr[d]);
                    bounds[3 + d] = Math.max(-1, maxWorld[d] - cropMinArr[d]);
                }
            }
            return bounds;
        } catch (final Exception e) {
            return new long[] { 1, 1, 1, 0, 0, 0 };
        }
    }
}
