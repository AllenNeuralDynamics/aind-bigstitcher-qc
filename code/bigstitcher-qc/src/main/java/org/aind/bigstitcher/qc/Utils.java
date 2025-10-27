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
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.util.Intervals;
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
}
