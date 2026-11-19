package org.aind.bigstitcher.qc;

import com.amazonaws.ClientConfiguration;
import com.amazonaws.auth.DefaultAWSCredentialsProviderChain;
import com.amazonaws.client.builder.AwsClientBuilder.EndpointConfiguration;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3URI;
import com.google.gson.GsonBuilder;
import java.net.URI;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import org.janelia.saalfeldlab.n5.KeyValueAccess;
import org.janelia.saalfeldlab.n5.N5Exception;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.s3.AmazonS3KeyValueAccess;
import org.janelia.saalfeldlab.n5.s3.AmazonS3Utils;
import org.janelia.saalfeldlab.n5.universe.N5Factory;
import org.janelia.saalfeldlab.n5.universe.StorageFormat;
import org.janelia.saalfeldlab.n5.universe.metadata.ome.ngff.v04.coordinateTransformations.CoordinateTransformation;
import org.janelia.saalfeldlab.n5.universe.metadata.ome.ngff.v04.coordinateTransformations.CoordinateTransformationAdapter;
import util.URITools;

/**
 * {@link N5Factory} for writing the OME-Zarr output to S3. It is configured like the stock
 * factory used by {@code URITools.instantiateN5Writer}, but sends far fewer requests and keeps
 * the AWS SDK's per-request retries from being cut short:
 * <ul>
 * <li>For every block, n5-zarr calls {@code createDirectories(parent)}, and n5-aws-s3 implements
 * that as one zero-byte PUT per ancestor prefix. For {@code .../overlay/0/0/0/8/20/21} that is 9
 * "directory" PUTs in addition to the block itself, repeated for every block.
 * {@link DirectoryCachingS3KeyValueAccess} writes the markers for a given parent only once.</li>
 * <li>The AWS SDK shares one retry budget across all threads using a client ("throttled
 * retries"). When many writer threads hit connection errors at about the same time, the budget
 * runs out and the SDK fails requests on the first error, regardless of AWS_MAX_ATTEMPTS. The
 * budget is disabled here; per-request retries still follow AWS_MAX_ATTEMPTS / AWS_RETRY_MODE.</li>
 * </ul>
 */
final class ResilientS3N5Factory extends N5Factory {

    private final String region;
    private final String endpoint;

    /**
     * @param region   AWS region name, or null for the SDK default
     * @param endpoint S3-compatible endpoint URL, or null for AWS S3
     */
    ResilientS3N5Factory(final String region, final String endpoint) {
        this.region = region;
        this.endpoint = endpoint;
    }

    /**
     * Opens a Zarr writer for the container. Local paths use the same writer as before; other
     * locations (s3://...) use this factory.
     */
    static N5Writer openZarrWriter(final URI containerUri) {
        if (URITools.isFile(containerUri))
            return URITools.instantiateN5Writer(StorageFormat.ZARR, containerUri);
        return forWriting(null).openWriter(StorageFormat.ZARR, containerUri);
    }

    /**
     * Factory with the same Zarr/Gson settings as {@code URITools.instantiateN5Writer}. Pass a
     * null endpoint for AWS S3, or an S3-compatible endpoint URL for testing.
     */
    static N5Factory forWriting(final String endpoint) {
        return new ResilientS3N5Factory(URITools.s3Region, endpoint)
                .zarrDimensionSeparator("/")
                .gsonBuilder(new GsonBuilder().registerTypeAdapter(
                        CoordinateTransformation.class,
                        new CoordinateTransformationAdapter()));
    }

    /**
     * Builds the client here because n5-aws-s3 4.3.0's
     * {@code AmazonS3Utils.createS3(String, String, AWSCredentialsProvider, ClientConfiguration, String)}
     * drops its ClientConfiguration argument, so {@link N5Factory#s3ClientConfiguration} has no effect.
     */
    @Override
    protected AmazonS3 createS3(final String uri) {
        try {
            final AmazonS3URI s3Uri = new AmazonS3URI(uri);
            return AmazonS3Utils.createS3(
                    s3Uri.getBucket(),
                    new DefaultAWSCredentialsProviderChain(),
                    endpoint == null ? null : new EndpointConfiguration(endpoint, region),
                    new ClientConfiguration().withThrottledRetries(false),
                    AmazonS3Utils.getS3Region(s3Uri, region));
        } catch (final RuntimeException e) {
            throw new N5Exception("Could not create s3 client from uri: " + uri, e);
        }
    }

    @Override
    public KeyValueAccess getKeyValueAccess(final URI uri) {
        final String scheme = uri.getScheme();
        if (scheme != null && AmazonS3Utils.S3_SCHEME.matcher(scheme).matches())
            return new DirectoryCachingS3KeyValueAccess(createS3(uri.toString()), uri);
        return super.getKeyValueAccess(uri);
    }

    /**
     * {@link AmazonS3KeyValueAccess} that remembers which directories it has created, so their
     * zero-byte marker objects are written once per parent instead of once per block. Threads that
     * need a directory another thread is still creating wait for it rather than writing the same
     * markers again.
     */
    static final class DirectoryCachingS3KeyValueAccess extends AmazonS3KeyValueAccess {

        private final ConcurrentHashMap<String, CompletableFuture<Void>> createdDirectories = new ConcurrentHashMap<>();

        DirectoryCachingS3KeyValueAccess(final AmazonS3 s3, final URI containerUri) {
            // createBucket=true matches the stock N5Factory behavior.
            super(s3, containerUri, true);
        }

        @Override
        public void createDirectories(final String normalPath) {
            final CompletableFuture<Void> created = new CompletableFuture<>();
            final CompletableFuture<Void> existing = createdDirectories.putIfAbsent(normalPath, created);
            if (existing != null) {
                existing.join(); // rethrows (wrapped) if the creating thread failed
                return;
            }
            try {
                super.createDirectories(normalPath);
                created.complete(null);
            } catch (final RuntimeException e) {
                // Forget the failed attempt so a retry creates the directory again.
                createdDirectories.remove(normalPath, created);
                created.completeExceptionally(e);
                throw e;
            }
        }

        @Override
        public void delete(final String normalPath) {
            try {
                super.delete(normalPath);
            } finally {
                // Deleted markers would otherwise still be considered present.
                createdDirectories.clear();
            }
        }
    }
}
