#!/bin/bash
# Checks that the local environment has everything this project needs.
# Run: bash check_setup.sh

check() {
  name="$1"
  cmd="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    version=$($cmd --version 2>&1 | head -n 1)
    echo "[OK]      $name -> $version"
  else
    echo "[MISSING] $name"
  fi
}

echo "=== Base tools ==="
check "Homebrew" "brew"
check "Git" "git"
check "Docker CLI" "docker"
check "Python 3" "python3"
check "pip3" "pip3"
check "Java" "java"

echo
echo "=== Docker daemon ==="
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    echo "[OK]      Docker Desktop is running"
  else
    echo "[MISSING] Docker is installed, but not running (open Docker Desktop)"
  fi
else
  echo "[MISSING] Docker is not installed"
fi

echo
echo "=== Optional (needed for RAG) ==="
check "Ollama" "ollama"

echo
echo "=== Python packages (if the venv is already created and activated) ==="
for pkg in pyspark sentence_transformers elasticsearch kafka jupyter; do
  if python3 -c "import $pkg" >/dev/null 2>&1; then
    echo "[OK]      python: $pkg"
  else
    echo "[MISSING] python: $pkg"
  fi
done

echo
echo "=== Done. What to do about MISSING items ==="
echo "Homebrew:            /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
echo "Docker Desktop:      brew install --cask docker"
echo "Git:                 brew install git"
echo "Java (for Spark):    brew install openjdk@17"
echo "Ollama:              brew install --cask ollama"
echo "Python packages:     pip3 install pyspark sentence-transformers elasticsearch kafka-python jupyter"
