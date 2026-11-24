/**
 * Vercel build step. Writes web/static/config.js from SIGNVISION_API_ORIGIN.
 * Example: https://api.yourdomain.com  (https, no trailing slash)
 * Unset or empty keeps same-origin requests, which is what local development uses.
 */
const fs = require("fs");
const path = require("path");

const origin = String(process.env.SIGNVISION_API_ORIGIN || "").trim().replace(/\/$/, "");
const file = path.join(__dirname, "static", "config.js");
const body = `window.SIGNVISION_API_ORIGIN = ${JSON.stringify(origin)};\n`;

fs.writeFileSync(file, body);
console.log(origin ? `API origin set to ${origin}` : "API origin empty; browser will use this page's host");
