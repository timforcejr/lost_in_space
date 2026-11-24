// Writes poster/fig/qr_live.png for the live page URL (error correction M, quiet zone 4 modules). Run: node poster/make_qr.js
const path = require("path");
const QRCode = require("qrcode");
const LIVE_URL = "https://timforcejr.github.io/lost_in_space/";
const out = path.join(__dirname, "fig", "qr_live.png");
QRCode.toFile(out, LIVE_URL, { errorCorrectionLevel: "M", margin: 4, scale: 40, type: "png" })
  .then(() => console.log("wrote " + out + " for " + LIVE_URL))
  .catch((e) => { console.error(e); process.exit(1); });
module.exports = { LIVE_URL };
