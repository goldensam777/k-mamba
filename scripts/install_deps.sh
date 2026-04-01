#!/usr/bin/env bash
# install_deps.sh — installe les dépendances de build pour k-mamba
# Requiert : Ubuntu 22.04+ / Debian, accès sudo
set -e

echo "==> Vérification des outils requis..."

MISSING=()

check_version() {
    local cmd="$1" min="$2" actual
    if ! actual=$($cmd --version 2>/dev/null | head -1); then
        MISSING+=("$cmd")
        return 1
    fi
    echo "    $cmd : $actual"
}

check_version gcc    "11"
check_version nasm   "2.15"
check_version cmake  "3.18"

if [ ${#MISSING[@]} -gt 0 ] || ! pkg-config --exists openblas 2>/dev/null && \
   ! [ -f /usr/lib/x86_64-linux-gnu/libopenblas.so ]; then
    echo "==> Installation des paquets manquants..."
    sudo apt-get update -qq
    # nasm + openblas (avec fallback --fix-missing si libgfortran5 indisponible)
    sudo apt-get install -y --fix-missing \
        nasm \
        libopenblas-dev \
        libopenblas-pthread-dev \
        build-essential \
        cmake \
        libgomp1 || true

    # Fallback : libgfortran5 depuis noble/main si noble-updates manquant
    if ! dpkg -l libopenblas-pthread-dev 2>/dev/null | grep -q '^ii'; then
        echo "==> Fallback : installation manuelle d'OpenBLAS..."
        TMP=$(mktemp -d)
        cd "$TMP"
        apt-get download \
            "libgfortran5=14-20240412-0ubuntu1" \
            libopenblas0-pthread \
            libopenblas-pthread-dev \
            libopenblas-dev 2>/dev/null || true
        for deb in *.deb; do
            sudo dpkg --ignore-depends=gcc-14-base -i "$deb" || \
            sudo dpkg -i "$deb" || true
        done
        cd - && rm -rf "$TMP"
    fi
fi

echo ""
echo "==> Résumé des versions installées :"
gcc     --version | head -1
nasm    --version | head -1
cmake   --version | head -1
pkg-config --modversion openblas 2>/dev/null || \
    dpkg -l | grep -E "libopenblas" | awk '{print $2, $3}' | head -3

echo ""
echo "==> Initialisation du submodule optimatrix..."
cd "$(git rev-parse --show-toplevel)"
git submodule update --init --recursive

echo ""
echo "==> Build de vérification..."
cmake -B build -DKMAMBA_BUILD_TESTS=OFF -DKMAMBA_BUILD_CUDA=OFF -Wno-dev -q
cmake --build build -j"$(nproc)" 2>&1 | tail -5

echo ""
echo "Toutes les dépendances sont prêtes."
