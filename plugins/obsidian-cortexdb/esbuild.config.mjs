// Bundles src/ into the one main.js Obsidian loads. The gRPC client comes
// from ../dsh-cortexdb-memory and is bundled in; Obsidian and Node's own
// modules are left for Obsidian's desktop runtime to provide.
import { builtinModules } from "node:module";
import { build } from "esbuild";

await build({
  entryPoints: ["src/main.js"],
  bundle: true,
  format: "cjs",
  platform: "node",
  target: "es2022",
  outfile: "main.js",
  external: ["obsidian", "electron", ...builtinModules, ...builtinModules.map((m) => `node:${m}`)],
  // The client uses Buffer, which Obsidian's window does not promise.
  banner: { js: 'var Buffer = require("buffer").Buffer;' },
  logLevel: "info",
});
