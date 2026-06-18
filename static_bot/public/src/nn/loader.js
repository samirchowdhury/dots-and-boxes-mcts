export async function loadBot(latestUrl) {
  const latest = await fetchJson(latestUrl);
  const manifestUrl = new URL(latest.manifest, latestUrl);
  const manifest = await fetchJson(manifestUrl);
  const weightsUrl = new URL(manifest.weightsFile, manifestUrl);
  const response = await fetch(weightsUrl);
  if (!response.ok) {
    throw new Error(`Could not load weights: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== manifest.weightsByteLength) {
    throw new Error(
      `Weights length mismatch: manifest=${manifest.weightsByteLength} actual=${buffer.byteLength}`,
    );
  }

  const weights = {};
  for (const [name, spec] of Object.entries(manifest.weights)) {
    weights[name] = new Float32Array(buffer, spec.byteOffset, spec.byteLength / 4);
  }

  return {
    latest,
    manifest,
    weights,
    weightsUrl: weightsUrl.toString(),
    manifestUrl: manifestUrl.toString(),
  };
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load ${url}: ${response.status} ${response.statusText}`);
  }
  return response.json();
}
