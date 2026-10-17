// The live view's 3D stack in one script: the graph, the three it renders
// with, and the bloom pass, all from that one three. Loaded as a classic
// script before the page's own; it only sets globals.
import ForceGraph3D from "3d-force-graph";
import { Vector2 } from "three";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

window.ForceGraph3D = ForceGraph3D;
window.CortexDBGraph3D = { Vector2, UnrealBloomPass };
