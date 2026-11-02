// Usage: node verify.mjs [ref.npy]   (default ./ref_orig.npy)
import { readFileSync } from "node:fs";
import { loadModel, scoreRow } from "./score.js";

const refPath = process.argv[2] ||
  "./ref_orig.npy";

function readNpyF32(path) {
  const b = readFileSync(path);
  if (b.toString("latin1", 1, 6) !== "NUMPY") throw new Error("not an npy file");
  const major = b[6];
  const hlen = major === 1 ? b.readUInt16LE(8) : b.readUInt32LE(8);
  const start = (major === 1 ? 10 : 12) + hlen;
  const header = b.toString("latin1", start - hlen, start);
  if (!header.includes("'<f4'") || header.includes("'fortran_order': True")) throw new Error("unexpected npy header: " + header);
  const n = (b.length - start) / 4;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = b.readFloatLE(start + 4 * i);
  return out;
}

const model = loadModel(JSON.parse(readFileSync("model.json", "utf8")));
const meta = JSON.parse(readFileSync("threshold.json", "utf8"));
const lines = readFileSync("sample.csv", "utf8").trim().split(/\r?\n/);
const cols = lines[0].split(",");
const idx = meta.features.map((f) => cols.indexOf(f));
if (idx.includes(-1)) throw new Error("feature missing from sample.csv");
const rows = lines.slice(1).map((l) => l.split(",").map(Number));
const ref = readNpyF32(refPath);
if (ref.length !== rows.length) throw new Error(`ref has ${ref.length} rows, csv has ${rows.length}`);

let maxDiff = 0, maxRow = -1, mismatches = 0;
rows.forEach((r, i) => {
  const { prob } = scoreRow(model, idx.map((j) => r[j]));
  const d = Math.abs(prob - ref[i]);
  if (d > maxDiff) { maxDiff = d; maxRow = i; }
  if ((prob >= meta.threshold) !== (ref[i] >= meta.threshold)) mismatches++;
});
console.log(`trees=${model.trees.length} rows=${rows.length} threshold=${meta.threshold} rule=prob>=threshold`);
console.log(`max abs diff = ${maxDiff.toExponential(3)} at row ${maxRow} (ref ${ref[maxRow]})`);
console.log(`decision mismatches = ${mismatches}`);
if (maxDiff > 1e-6 || mismatches > 0) { console.log("FAIL"); process.exit(1); }
console.log("PASS");
