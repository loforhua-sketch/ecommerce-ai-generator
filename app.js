const fields = {
  productName: document.querySelector("#productName"),
  category: document.querySelector("#category"),
  price: document.querySelector("#price"),
  originPrice: document.querySelector("#originPrice"),
  imageUrl: document.querySelector("#imageUrl"),
  audience: document.querySelector("#audience"),
  features: document.querySelector("#features"),
  specs: document.querySelector("#specs"),
};

const preview = {
  page: document.querySelector("#detailPage"),
  category: document.querySelector("#previewCategory"),
  name: document.querySelector("#previewName"),
  tagline: document.querySelector("#previewTagline"),
  price: document.querySelector("#previewPrice"),
  origin: document.querySelector("#previewOrigin"),
  image: document.querySelector("#previewImage"),
  features: document.querySelector("#previewFeatures"),
  specs: document.querySelector("#previewSpecs"),
  faq: document.querySelector("#previewFaq"),
  storyTitle: document.querySelector("#storyTitle"),
  storyText: document.querySelector("#storyText"),
  quality: document.querySelector("#qualityList"),
  status: document.querySelector("#statusText"),
};

const demo = {
  productName: "AeroFit 智能恒温运动水杯",
  category: "智能运动装备",
  price: "199",
  originPrice: "299",
  imageUrl:
    "https://images.unsplash.com/photo-1602143407151-7111542de6e8?auto=format&fit=crop&w=1200&q=80",
  audience: "通勤、健身、户外运动用户",
  features: "6 小时智能恒温\n316 不锈钢内胆\nAPP 饮水提醒\n一键拆洗杯盖",
  specs: "容量：650ml\n重量：340g\n材质：316 不锈钢\n续航：14 天",
};

function getLines(value) {
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function parseSpecs(value) {
  return getLines(value).map((line) => {
    const separator = line.includes("：") ? "：" : ":";
    const [key, ...rest] = line.split(separator);
    return {
      key: key?.trim() || "参数",
      value: rest.join(separator).trim() || line,
    };
  });
}

function formatPrice(value) {
  const number = Number(value || 0);
  return `¥${Number.isFinite(number) && number > 0 ? number : 0}`;
}

function featureDescription(feature, category) {
  const map = [
    `围绕“${feature}”放大真实使用价值，降低用户理解成本。`,
    `适合${category}场景，帮助买家快速判断是否匹配需求。`,
    `详情页自动转成利益点表达，适合首屏和短视频落地页复用。`,
    `从参数转换为购买理由，让商品优势更容易被记住。`,
  ];
  return map[Math.floor(feature.length % map.length)];
}

function buildFaq(name, audience) {
  return [
    {
      q: `${name}适合哪些人？`,
      a: `适合${audience || "有明确品质需求的用户"}，页面文案会优先突出高频场景和决策理由。`,
    },
    {
      q: "详情页内容可以继续编辑吗？",
      a: "可以。左侧字段变更后右侧会实时同步，导出前仍可调整标题、卖点和参数。",
    },
    {
      q: "这个 MVP 是否接入真实 AI？",
      a: "当前版本使用本地规则生成文案结构，后续可以替换为大模型接口生成更细的图文脚本。",
    },
  ];
}

function renderQuality(state) {
  const featureCount = getLines(state.features).length;
  const specCount = parseSpecs(state.specs).length;
  const checks = [
    { label: "商品名称已填写", pass: state.productName.trim().length >= 2 },
    { label: "至少 3 个核心卖点", pass: featureCount >= 3 },
    { label: "至少 3 个商品参数", pass: specCount >= 3 },
    { label: "主图 URL 已填写", pass: state.imageUrl.trim().length > 0 },
    { label: "售价低于原价", pass: Number(state.price) < Number(state.originPrice) },
  ];

  preview.quality.innerHTML = checks
    .map(
      (check) => `
        <div class="quality-item ${check.pass ? "pass" : ""}">
          <span class="quality-dot"></span>
          <span>${check.label}</span>
        </div>
      `,
    )
    .join("");
}

function readState() {
  return Object.fromEntries(
    Object.entries(fields).map(([key, element]) => [key, element.value]),
  );
}

function render() {
  const state = readState();
  const name = state.productName.trim() || "未命名商品";
  const category = state.category;
  const audience = state.audience.trim() || "目标用户";
  const features = getLines(state.features);
  const specs = parseSpecs(state.specs);

  preview.category.textContent = category;
  preview.name.textContent = name;
  preview.tagline.textContent = `为${audience}打造，兼顾颜值、性能与日常效率。`;
  preview.price.textContent = formatPrice(state.price);
  preview.origin.textContent = formatPrice(state.originPrice);
  preview.image.src = state.imageUrl.trim();
  preview.image.alt = name;
  preview.storyTitle.textContent = `让${category}的购买理由一屏说清`;
  preview.storyText.textContent = `${name}的详情页已根据商品信息自动组织为首屏利益点、核心卖点、参数表和购买 FAQ，适合快速搭建可投放页面。`;

  preview.features.innerHTML = features
    .map(
      (feature) => `
        <div class="feature-card">
          <strong>${feature}</strong>
          <p>${featureDescription(feature, category)}</p>
        </div>
      `,
    )
    .join("");

  preview.specs.innerHTML = specs
    .map(
      (spec) => `
        <div class="spec-row">
          <dt>${spec.key}</dt>
          <dd>${spec.value}</dd>
        </div>
      `,
    )
    .join("");

  preview.faq.innerHTML = buildFaq(name, audience)
    .map(
      (item) => `
        <div class="faq-item">
          <strong>${item.q}</strong>
          <p>${item.a}</p>
        </div>
      `,
    )
    .join("");

  renderQuality(state);
  preview.status.textContent = `已同步 ${new Date().toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  })}`;
}

function generateCopy() {
  const state = readState();
  const category = state.category;
  const audience = state.audience || "目标用户";
  const generated = [
    `专为${audience}设计`,
    `${category}场景快速上手`,
    "高颜值外观提升下单信心",
    "核心参数可视化展示",
  ];

  fields.features.value = Array.from(new Set([...getLines(state.features), ...generated]))
    .slice(0, 6)
    .join("\n");
  render();
}

function loadDemo() {
  Object.entries(demo).forEach(([key, value]) => {
    fields[key].value = value;
  });
  render();
}

function exportHtml() {
  const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${preview.name.textContent}</title>
  <link rel="stylesheet" href="./styles.css" />
</head>
<body>
  ${preview.page.outerHTML}
</body>
</html>`;

  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${fields.productName.value || "detail-page"}.html`;
  anchor.click();
  URL.revokeObjectURL(url);
}

document.querySelector("#generateCopy").addEventListener("click", generateCopy);
document.querySelector("#loadDemo").addEventListener("click", loadDemo);
document.querySelector("#exportHtml").addEventListener("click", exportHtml);

document.querySelectorAll(".device-toggle button").forEach((button) => {
  button.addEventListener("click", () => {
    document
      .querySelectorAll(".device-toggle button")
      .forEach((item) => item.classList.remove("toggle-active"));
    button.classList.add("toggle-active");
    preview.page.classList.toggle("mobile-preview", button.dataset.width === "mobile");
    preview.page.classList.toggle("desktop-preview", button.dataset.width === "desktop");
  });
});

Object.values(fields).forEach((field) => {
  field.addEventListener("input", render);
  field.addEventListener("change", render);
});

preview.image.addEventListener("error", () => {
  preview.image.removeAttribute("src");
  preview.image.alt = "图片加载失败，请检查商品图片 URL";
});

render();
