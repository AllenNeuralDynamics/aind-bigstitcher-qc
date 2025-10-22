# hash:sha256:c2ac09b3a234182826fe903d6596e982cf5c5f6e402bea2600b5ccc1812a9dd2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

# ---- Install Azul Zulu JDK 8u462 (with JavaFX) ----
ARG AZUL_JDK_URL="https://cdn.azul.com/zulu/bin/zulu8.88.0.19-ca-fx-jdk8.0.462-linux_x64.tar.gz"
ENV JAVA_HOME=/usr/lib/jvm/zulu-8
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN apt-get update \
 && apt-get install -y --no-install-recommends libblosc1 curl jq ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 # Download JDK
 && curl -fsSL "${AZUL_JDK_URL}" -o /tmp/zulu.tar.gz \
 # Determine extracted folder name from the tarball
 && ZULU_DIR="$(tar -tzf /tmp/zulu.tar.gz | head -1 | cut -f1 -d'/')" \
 # Extract to /usr/lib/jvm and create a stable symlink at $JAVA_HOME
 && mkdir -p /usr/lib/jvm \
 && tar -xzf /tmp/zulu.tar.gz -C /usr/lib/jvm \
 && ln -s "/usr/lib/jvm/${ZULU_DIR}" "${JAVA_HOME}" \
 # Register java/javac with update-alternatives
 && update-alternatives --install /usr/bin/java  java  "${JAVA_HOME}/bin/java"  20000 \
 && update-alternatives --install /usr/bin/javac javac "${JAVA_HOME}/bin/javac" 20000 \
 # Cleanup
 && rm -f /tmp/zulu.tar.gz

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
