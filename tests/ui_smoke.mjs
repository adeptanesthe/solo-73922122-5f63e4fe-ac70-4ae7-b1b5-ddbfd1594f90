/* Headless smoke test of the UI against a live server using jsdom.
 *
 * Usage: node tests/ui_smoke.mjs http://127.0.0.1:8123
 * Requires jsdom available from the working directory (see README).
 */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

// jsdom is an optional dev dependency; point JSDOM_DIR at a node_modules
// directory that contains it when it is not installed next to this file.
const require = createRequire(
  process.env.JSDOM_DIR ? path.join(process.env.JSDOM_DIR, "noop.js") : import.meta.url,
);
const { JSDOM, VirtualConsole } = require("jsdom");

const BASE = process.argv[2] || "http://127.0.0.1:8123";
const root = path.resolve(new URL("..", import.meta.url).pathname);

const html = fs.readFileSync(path.join(root, "app/static/index.html"), "utf8");
const script = fs.readFileSync(path.join(root, "app/static/app.js"), "utf8");

const virtualConsole = new VirtualConsole();
const errors = [];
virtualConsole.on("jsdomError", (e) => errors.push(String(e)));
virtualConsole.on("error", (e) => errors.push(String(e)));

const dom = new JSDOM(html, {
  url: BASE + "/",
  runScripts: "outside-only",
  virtualConsole,
  pretendToBeVisual: true,
});
const { window } = dom;
window.fetch = (input, init) => fetch(new URL(input, BASE), init);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, label, timeout = 20000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (fn()) return;
    await sleep(50);
  }
  throw new Error("timeout waiting for " + label);
}
const $ = (id) => window.document.getElementById(id);

window.eval(script);
await sleep(200);

const results = [];
function check(name, condition, detail = "") {
  results.push({ name, ok: !!condition, detail });
  console.log(`${condition ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
}

// 1. samples are listed
await until(() => $("sample-buttons").children.length > 0, "sample buttons");
check("样例按钮已渲染", $("sample-buttons").children.length >= 3,
  $("sample-buttons").textContent.trim());

// 2. load the repeat sample
const repeatBtn = [...$("sample-buttons").children].find((b) => b.textContent.includes("sample_repeat"));
repeatBtn.click();
await until(() => !$("workspace").hidden && $("progress-text").textContent !== "0 / 0 帧", "sample loaded");
check("载入样例后显示工作区", !$("workspace").hidden, $("progress-text").textContent);
check("初始无候选字幕", $("confirmed-text").textContent === "", `“${$("confirmed-text").textContent}”`);

// 3. advance one chunk
window.document.getElementById("chunk-size").value = "5";
$("chunk-size").dispatchEvent(new window.Event("input"));
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("5 /"), "advance chunk");
check("分块推进更新进度", $("progress-text").textContent.startsWith("5 /"), $("progress-text").textContent);
check("候选表非空", $("beam-body").children.length > 0, `${$("beam-body").children.length} 束`);
const firstText = $("beam-body").children[0].querySelector(".beam-text").textContent;
check("最佳候选含字符", firstText.length > 0, firstText);

// 4. confirm the best candidate
$("confirm-best").click();
await until(() => $("confirmed-text").textContent.length > 0, "confirm");
const confirmed = $("confirmed-text").textContent;
check("确认后前缀显示", confirmed.length > 0, confirmed);
check("blank 分数显示", $("ps-blank").textContent !== "—", $("ps-blank").textContent);
check("非 blank 分数显示", $("ps-nonblank").textContent !== "—", $("ps-nonblank").textContent);
check("撤销按钮可用", !$("undo-btn").disabled);

// 5. advance after confirmation, then undo
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("10 /"), "advance after confirm");
$("undo-btn").click();
await until(() => $("confirmed-text").textContent === "", "undo");
check("撤销后前缀清空", $("confirmed-text").textContent === "");
check("撤销后进度保留", $("progress-text").textContent.startsWith("10 /"), $("progress-text").textContent);
check("撤销后候选非空", $("beam-body").children.length > 0);

// 6. failure keeps state: confirm text not in beam
$("prefix-input").value = "zzzz";
$("apply-prefix").click();
await until(() => !$("op-error").hidden, "error shown");
const beforeProgress = $("progress-text").textContent;
check("非法确认显示错误", !$("op-error").hidden, $("op-error").textContent);
check("错误后进度不变", $("progress-text").textContent === beforeProgress, $("progress-text").textContent);

// 7. historical frame inspection
$("peek-frame").value = "2";
$("peek-frame").dispatchEvent(new window.Event("input"));
await sleep(400);
check("历史帧提示只读", $("peek-note").textContent.includes("历史回放"), $("peek-note").textContent);
check("历史帧概率已渲染", $("frame-probs").children.length > 0);

// 8. continuous run to the end
$("peek-frame").value = $("peek-frame").max;
$("peek-frame").dispatchEvent(new window.Event("input"));
$("run-btn").click();
await until(() => $("progress-text").textContent.startsWith("15 / 15"), "run to end", 30000);
check("连续推进到结尾", $("progress-text").textContent.startsWith("15 / 15"), $("progress-text").textContent);
await until(() => $("stop-btn").hidden, "run loop finished");
check("运行结束停止按钮隐藏", $("stop-btn").hidden);

// 9. zero-probability sample shows -∞ without NaN
const zeroBtn = [...$("sample-buttons").children].find((b) => b.textContent.includes("sparse_zero"));
zeroBtn.click();
await until(() => $("progress-text").textContent.startsWith("0 / 7"), "zero sample loaded");
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("7 / 7") || $("progress-text").textContent.startsWith("5 / 7"), "zero sample advance");
await sleep(300);
const bodyText = $("beam-body").textContent;
check("零概率样例无 NaN", !bodyText.includes("NaN"), bodyText.slice(0, 120));
check("零概率样例显示 -∞", bodyText.includes("-∞") || $("frame-probs").textContent.includes("0.000"));

// 10. no uncaught UI errors
check("无未捕获脚本错误", errors.length === 0, errors.join(" | ").slice(0, 200));

// 11. local frame revision panel
const repeatBtn2 = [...$("sample-buttons").children].find((b) => b.textContent.includes("sample_repeat"));
repeatBtn2.click();
await until(() => $("progress-text").textContent.startsWith("0 / 15"), "reload repeat sample");
$("chunk-size").value = "4";
$("chunk-size").dispatchEvent(new window.Event("input"));
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("4 /"), "advance 4 frames");
$("confirm-best").click();
await until(() => $("confirmed-text").textContent.length > 0, "confirm before revision");
const confirmedText = $("confirmed-text").textContent;

// 11a. grid geometry: 1 row initially, grows with the frame-count slider
check("修订网格初始 1 行", $("revise-grid").querySelectorAll("tbody tr").length === 1);
$("revise-count").value = "3";
$("revise-count").dispatchEvent(new window.Event("input"));
check("修订网格随帧数变为 3 行", $("revise-grid").querySelectorAll("tbody tr").length === 3);
check("修订网格列数等于词表", $("revise-grid").querySelectorAll("tbody tr")[0].querySelectorAll("input").length === 4);

// 11b. prefill reads the live probabilities
$("revise-start").value = "1";
$("revise-start").dispatchEvent(new window.Event("input"));
$("revise-prefill").click();
await until(
  () => [...$("revise-grid").querySelectorAll("tbody tr")[0].querySelectorAll("input")]
    .every((i) => i.value !== ""),
  "prefill populated",
);
const firstInputs = [...$("revise-grid").querySelectorAll("tbody tr")[0].querySelectorAll("input")];
check("预填概率已写入", firstInputs.every((i) => Number(i.value) >= 0), firstInputs.map((i) => i.value).join(","));

// 11c. identity revision preserves the confirmed text and records history
$("revise-submit").click();
await until(() => $("revise-list").children.length === 1, "revision applied");
check("修订后确认文本保留", $("confirmed-text").textContent === confirmedText, $("confirmed-text").textContent);
check("修订记录渲染", $("revise-list").children[0].textContent.includes("帧 1～3"),
  $("revise-list").children[0].textContent);
check("修订后候选非空", $("beam-body").children.length > 0);
check("修订后无错误提示", $("revise-error").hidden);

// 11d. invalid rows are blocked client-side and show an error
const badInput = $("revise-grid").querySelectorAll("tbody input")[0];
badInput.value = "1.7";
$("revise-submit").click();
await until(() => !$("revise-error").hidden, "client validation error");
check("越界概率被前端拦截", $("revise-error").textContent.includes("标红"));
check("非法单元格标红", badInput.classList.contains("invalid"));

// 11e. normalize helper makes each row sum to 1
badInput.value = "0.2";
$("revise-normalize").click();
const normalized = [...$("revise-grid").querySelectorAll("tbody tr")[0].querySelectorAll("input")]
  .map((i) => Number(i.value));
const sum = normalized.reduce((acc, v) => acc + v, 0);
check("归一化后行和约为 1", Math.abs(sum - 1) < 0.01, String(sum));

// 11f. a failing server replay surfaces the earliest failure and keeps state.
// Four blank-only frames 0..3 make the frame-4 confirmation of "a" unreachable.
$("revise-start").value = "0";
$("revise-start").dispatchEvent(new window.Event("input"));
$("revise-count").value = "4";
$("revise-count").dispatchEvent(new window.Event("input"));
check("失败场景网格为 4 行", $("revise-grid").querySelectorAll("tbody tr").length === 4);
const allRows = [...$("revise-grid").querySelectorAll("tbody input")];
check("失败场景单元格数为 4×词表", allRows.length === 16, String(allRows.length));
// blank-only frames cannot support any non-empty confirmed prefix
allRows.forEach((input, idx) => { input.value = idx % 4 === 0 ? "1" : "0"; });
const progressBefore = $("progress-text").textContent;
const confirmedBefore = $("confirmed-text").textContent;
$("revise-submit").click();
await until(() => !$("revise-error").hidden && $("revise-error").textContent.includes("第 1 次失败"),
  "server replay failure", 20000);
check("重演失败显示最早失败处", $("revise-error").textContent.includes("原会话已完整保留"),
  $("revise-error").textContent);
check("失败后进度保留", $("progress-text").textContent === progressBefore);
check("失败后确认文本保留", $("confirmed-text").textContent === confirmedBefore);
check("失败不新增修订记录", $("revise-list").children.length === 1);

// 11g. no uncaught errors after the whole revision workflow
check("修订流程无未捕获脚本错误", errors.length === 0, errors.join(" | ").slice(0, 200));

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
