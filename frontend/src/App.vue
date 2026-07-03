<template>
  <main class="app-shell">
    <aside class="left-panel">
      <div class="brand">
        <div class="brand-mark">AI</div>
        <div>
          <h1>AI 电商详情页生成器</h1>
          <p>上传商品图片，自动生成多平台标题、卖点文案和 HTML 详情页。</p>
        </div>
      </div>

      <el-alert
        v-if="formError"
        class="form-alert"
        type="warning"
        :title="formError"
        show-icon
        :closable="false"
      />

      <el-form label-position="top" class="generator-form" @submit.prevent="generate">
        <el-form-item label="商品图片" :error="fieldErrors.files">
          <el-upload
            v-model:file-list="fileList"
            drag
            multiple
            accept="image/*"
            :auto-upload="false"
            list-type="picture"
            @change="clearError('files')"
            @remove="clearError('files')"
          >
            <el-icon class="upload-icon"><UploadFilled /></el-icon>
            <div class="el-upload__text">拖拽图片到此处，或点击选择</div>
          </el-upload>
        </el-form-item>

        <el-form-item label="商品名称" :error="fieldErrors.productName">
          <el-input v-model.trim="form.productName" placeholder="请输入商品名称" @input="clearError('productName')" />
        </el-form-item>

        <div class="form-grid">
          <el-form-item label="品类" :error="fieldErrors.category">
            <el-input v-model.trim="form.category" placeholder="如：家居百货" @input="clearError('category')" />
          </el-form-item>
          <el-form-item label="人群" :error="fieldErrors.audience">
            <el-input v-model.trim="form.audience" placeholder="如：通勤白领" @input="clearError('audience')" />
          </el-form-item>
        </div>

        <div class="form-grid">
          <el-form-item label="售价" :error="fieldErrors.price">
            <el-input v-model.trim="form.price" placeholder="199" @input="clearError('price')" />
          </el-form-item>
          <el-form-item label="原价">
            <el-input v-model.trim="form.originPrice" placeholder="299" />
          </el-form-item>
        </div>

        <el-button native-type="submit" type="primary" size="large" :loading="loading">
          <el-icon><MagicStick /></el-icon>
          批量生成
        </el-button>

        <div v-if="loading" class="progress-box">
          <div class="loader"></div>
          <div class="progress-copy">
            <strong>{{ progressText }}</strong>
            <el-progress :percentage="progress" :stroke-width="8" />
          </div>
        </div>
      </el-form>
    </aside>

    <section class="result-panel">
      <div class="toolbar">
        <div>
          <h2>生成结果</h2>
          <span>{{ result ? "已渲染本次返回的第一条结果" : "等待上传商品图片" }}</span>
        </div>
        <el-button :icon="Refresh" @click="loadHistory">刷新历史</el-button>
      </div>

      <el-alert
        v-if="emptyResultMessage"
        class="form-alert"
        type="info"
        :title="emptyResultMessage"
        show-icon
        :closable="false"
      />

      <el-empty v-if="!result && !loading" description="暂无生成结果" />

      <article v-else-if="result" class="result-card">
        <div class="image-wrap">
          <img :src="imageSrc(result)" :alt="analysis.product_name || '商品图片'" />
        </div>

        <div class="card-body">
          <div class="card-head">
            <div>
              <h3>{{ analysis.product_name || form.productName || "未识别商品名称" }}</h3>
              <p>{{ analysis.category || form.category || "未识别品类" }} · {{ result.created_at || "刚刚生成" }}</p>
            </div>
            <el-tag v-if="result.id" type="success">#{{ result.id }}</el-tag>
          </div>

          <div class="info-grid overview-grid">
            <div>
              <strong>商品名称</strong>
              <span>{{ analysis.product_name || form.productName || "未识别" }}</span>
            </div>
            <div>
              <strong>品类</strong>
              <span>{{ analysis.category || form.category || "未识别" }}</span>
            </div>
            <div>
              <strong>人群</strong>
              <span>{{ analysis.audience || form.audience || "未识别" }}</span>
            </div>
          </div>

          <el-tabs>
            <el-tab-pane label="平台标题">
              <div class="title-list">
                <CopyLine label="淘宝标题" :value="analysis.taobao_title" />
                <CopyLine label="拼多多标题" :value="analysis.pdd_title" />
                <CopyLine label="抖店标题" :value="analysis.douyin_title" />
              </div>
            </el-tab-pane>

            <el-tab-pane label="卖点">
              <ul v-if="sellingPoints.length" class="selling-list">
                <li v-for="point in sellingPoints" :key="point">{{ point }}</li>
              </ul>
              <el-empty v-else description="暂无卖点" />
            </el-tab-pane>

            <el-tab-pane label="详情页 HTML 预览">
              <iframe :srcdoc="result.html || ''" title="详情页 HTML 预览"></iframe>
            </el-tab-pane>
          </el-tabs>

          <div class="card-actions">
            <el-button v-if="result.id" type="primary" :icon="Download" @click="download(result.id)">导出 HTML</el-button>
            <el-button :icon="DocumentCopy" @click="copyHtml(result.html)">复制 HTML</el-button>
          </div>
        </div>
      </article>
    </section>
  </main>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import { DocumentCopy, Download, MagicStick, Refresh, UploadFilled } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";
import CopyLine from "./components/CopyLine.vue";
import { api, exportUrl, generateDetails } from "./api/client";

const loading = ref(false);
const progress = ref(0);
const fileList = ref([]);
const result = ref(null);
const emptyResultMessage = ref("");
const formError = ref("");
const fieldErrors = reactive({
  files: "",
  productName: "",
  category: "",
  audience: "",
  price: ""
});
const form = reactive({
  productName: "",
  category: "",
  audience: "",
  price: "",
  originPrice: ""
});

let progressTimer = null;

const analysis = computed(() => result.value?.analysis || {});
const sellingPoints = computed(() => {
  const value = analysis.value.selling_points;
  return Array.isArray(value) ? value : [];
});

const progressText = computed(() => {
  if (progress.value < 25) return "正在上传商品图片";
  if (progress.value < 55) return "AI 正在识别商品信息";
  if (progress.value < 78) return "正在生成平台标题和卖点文案";
  if (progress.value < 95) return "正在组装 HTML 详情页";
  return "正在保存生成结果";
});

function imageSrc(item) {
  if (!item?.image_path) return "";
  const base = import.meta.env.VITE_API_BASE_URL || "";
  return `${base}/api/uploads/${item.image_path}`;
}

function clearError(field) {
  fieldErrors[field] = "";
  formError.value = "";
  emptyResultMessage.value = "";
}

function validateForm() {
  Object.keys(fieldErrors).forEach((key) => {
    fieldErrors[key] = "";
  });
  formError.value = "";

  if (!fileList.value.length) fieldErrors.files = "请先上传至少一张商品图片";
  if (!form.productName) fieldErrors.productName = "请输入商品名称";
  if (!form.category) fieldErrors.category = "请输入品类";
  if (!form.audience) fieldErrors.audience = "请输入目标人群";
  if (!form.price) fieldErrors.price = "请输入售价";

  const messages = Object.values(fieldErrors).filter(Boolean);
  if (messages.length) {
    formError.value = messages[0];
    ElMessage.warning(messages[0]);
    return false;
  }
  return true;
}

function startProgress() {
  progress.value = 8;
  progressTimer = window.setInterval(() => {
    if (progress.value < 92) {
      progress.value += progress.value < 60 ? 7 : 3;
    }
  }, 650);
}

function stopProgress(done = false) {
  if (progressTimer) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
  progress.value = done ? 100 : 0;
}

function buildPayload() {
  const payload = new FormData();
  fileList.value.forEach((file) => {
    if (file.raw) payload.append("files", file.raw);
  });
  payload.append("product_name", form.productName);
  payload.append("category", form.category);
  payload.append("audience", form.audience);
  payload.append("price", form.price);
  payload.append("origin_price", form.originPrice);
  return payload;
}

async function generate() {
  if (loading.value) return;
  if (!validateForm()) return;

  const payload = buildPayload();
  loading.value = true;
  result.value = null;
  emptyResultMessage.value = "";
  startProgress();

  try {
    const response = await generateDetails(payload);
    console.log(response.data);

    const items = Array.isArray(response.data?.items) ? response.data.items : [];
    if (!items.length) {
      emptyResultMessage.value = "后端返回 items 为空，未生成可展示结果。";
      ElMessage.warning(emptyResultMessage.value);
      return;
    }

    result.value = items[0];
    ElMessage.success("生成成功，已显示第一条结果");
  } catch (error) {
    console.error("[generate] 请求失败", error);
    const message =
      error.response?.data?.detail || error.message || "生成失败，请检查后端服务、Ollama 服务或接口配置";
    formError.value = message;
    ElMessage.error(message);
  } finally {
    stopProgress(Boolean(result.value));
    window.setTimeout(() => {
      loading.value = false;
      if (progress.value === 100) progress.value = 0;
    }, 500);
  }
}

async function loadHistory() {
  try {
    const { data } = await api.get("/api/generations");
    const items = Array.isArray(data.items) ? data.items : [];
    result.value = items[0] || null;
    emptyResultMessage.value = items.length ? "" : "历史记录为空。";
  } catch (error) {
    console.error("[history] 请求失败", error);
    ElMessage.error("加载历史记录失败，请检查后端服务");
  }
}

function download(id) {
  window.location.href = exportUrl(id);
}

async function copyHtml(html) {
  await navigator.clipboard.writeText(html || "");
  ElMessage.success("HTML 已复制");
}

onMounted(loadHistory);
</script>
