<template>
  <main class="app-shell">
    <aside class="left-panel">
      <div class="brand">
        <div class="brand-mark">AI</div>
        <div>
          <h1>AI 电商详情页生成器</h1>
          <p>一次上传多张商品图片，批量生成多平台标题、卖点和 HTML 详情页。</p>
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

        <div class="form-grid">
          <el-form-item label="平台选择">
            <el-select v-model="form.platform">
              <el-option label="淘宝" value="taobao" />
              <el-option label="拼多多" value="pdd" />
              <el-option label="抖店" value="douyin" />
            </el-select>
          </el-form-item>
          <el-form-item label="风格选择">
            <el-select v-model="form.style">
              <el-option label="简约" value="simple" />
              <el-option label="高端" value="premium" />
              <el-option label="促销" value="promotion" />
              <el-option label="场景化" value="scene" />
            </el-select>
          </el-form-item>
        </div>

        <el-button native-type="submit" type="primary" size="large" :loading="loading">
          <el-icon><MagicStick /></el-icon>
          {{ fileList.length > 1 ? "批量生成" : "生成详情页" }}
        </el-button>

        <div v-if="loading" class="progress-box">
          <div class="loader"></div>
          <div class="progress-copy">
            <strong>正在生成第 {{ currentIndex }} 张 / 共 {{ totalCount }} 张</strong>
            <span>{{ progressText }}</span>
            <el-progress :percentage="progress" :stroke-width="8" />
          </div>
        </div>
      </el-form>
    </aside>

    <section class="result-panel">
      <div class="toolbar">
        <div>
          <h2>生成结果</h2>
          <span>{{ toolbarText }}</span>
        </div>
        <div class="toolbar-actions">
          <el-button :icon="Refresh" @click="loadHistory">刷新历史</el-button>
          <el-button type="primary" :icon="Download" :disabled="!successfulResults.length" @click="downloadAllZip">
            下载全部 ZIP
          </el-button>
        </div>
      </div>

      <el-alert
        v-if="emptyResultMessage"
        class="form-alert"
        type="info"
        :title="emptyResultMessage"
        show-icon
        :closable="false"
      />

      <el-empty v-if="!results.length && !loading" description="暂无生成结果" />

      <div v-else class="result-grid">
        <article
          v-for="(item, index) in results"
          :key="item.id || `${item.filename}-${index}`"
          class="result-card"
          :class="{ failed: isFailed(item) }"
        >
          <template v-if="isFailed(item)">
            <div class="failed-thumb">
              <el-icon><WarningFilled /></el-icon>
            </div>
            <div class="card-body">
              <div class="card-head">
                <div>
                  <h3>{{ item.filename || `第 ${index + 1} 张图片` }}</h3>
                  <p>生成失败</p>
                </div>
                <el-tag type="danger">失败</el-tag>
              </div>
              <el-alert type="error" :title="item.error || '未知错误'" show-icon :closable="false" />
            </div>
          </template>

          <template v-else>
            <div class="image-wrap">
              <img :src="imageSrc(item)" :alt="item.analysis?.product_name || item.product_name || '商品图片'" />
            </div>

            <div class="card-body">
              <div class="card-head">
                <div>
                  <h3>{{ productName(item) }}</h3>
                  <p>{{ item.analysis?.category || item.category || "未识别品类" }} · {{ item.created_at || "刚刚生成" }}</p>
                </div>
                <el-tag v-if="item.id" type="success">#{{ item.id }}</el-tag>
              </div>

              <div class="info-grid overview-grid">
                <div>
                  <strong>商品名称</strong>
                  <span>{{ productName(item) }}</span>
                </div>
                <div>
                  <strong>图片文件</strong>
                  <span>{{ item.filename || item.image_path || "未命名图片" }}</span>
                </div>
              </div>

              <el-tabs>
                <el-tab-pane label="平台标题">
                  <div class="title-list">
                    <CopyLine label="淘宝标题" :value="item.analysis?.taobao_title" />
                    <CopyLine label="拼多多标题" :value="item.analysis?.pdd_title" />
                    <CopyLine label="抖店标题" :value="item.analysis?.douyin_title" />
                  </div>
                </el-tab-pane>

                <el-tab-pane label="卖点">
                  <ul v-if="sellingPoints(item).length" class="selling-list">
                    <li v-for="point in sellingPoints(item)" :key="point">{{ point }}</li>
                  </ul>
                  <el-empty v-else description="暂无卖点" />
                </el-tab-pane>

                <el-tab-pane label="HTML 详情页预览">
                  <iframe :srcdoc="previewHtml(item)" title="HTML 详情页预览"></iframe>
                </el-tab-pane>
              </el-tabs>

              <div class="card-actions">
                <el-button v-if="item.id" type="primary" :icon="Download" @click="download(item.id)">导出 HTML</el-button>
                <el-button :icon="DocumentCopy" @click="copyHtml(previewHtml(item))">复制 HTML</el-button>
              </div>
            </div>
          </template>
        </article>
      </div>
    </section>
  </main>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import { DocumentCopy, Download, MagicStick, Refresh, UploadFilled, WarningFilled } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";
import CopyLine from "./components/CopyLine.vue";
import { api, exportUrl, exportZipUrl, generateDetails } from "./api/client";

const loading = ref(false);
const progress = ref(0);
const currentIndex = ref(0);
const totalCount = ref(0);
const fileList = ref([]);
const results = ref([]);
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
  originPrice: "",
  platform: "taobao",
  style: "simple"
});

let progressTimer = null;

const successfulResults = computed(() => results.value.filter((item) => !isFailed(item) && item.id));
const toolbarText = computed(() => {
  if (!results.value.length) return "等待上传商品图片";
  const failedCount = results.value.filter(isFailed).length;
  return `共 ${results.value.length} 条结果，成功 ${successfulResults.value.length} 条，失败 ${failedCount} 条`;
});

const progressText = computed(() => {
  if (!totalCount.value) return "正在准备生成任务";
  return `后端将按图片逐张分析，失败图片会保留原因并继续处理后续图片`;
});

function isFailed(item) {
  return item?.status === "failed" || Boolean(item?.error);
}

function productName(item) {
  return item?.analysis?.product_name || item?.product_name || form.productName || "未识别商品名称";
}

function sellingPoints(item) {
  const analysis = item?.analysis || {};
  const value = analysis.selling_points || analysis.core_selling_points || [];
  return Array.isArray(value) ? value : [];
}

function imageSrc(item) {
  if (!item?.image_path) return "";
  const base = import.meta.env.VITE_API_BASE_URL || "";
  return `${base}/api/uploads/${item.image_path}`;
}

function previewHtml(item) {
  return item?.html_files?.[form.platform] || item?.html || "";
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
  progress.value = 5;
  currentIndex.value = 1;
  totalCount.value = fileList.value.length;
  progressTimer = window.setInterval(() => {
    if (progress.value < 92) {
      progress.value += progress.value < 60 ? 5 : 2;
      currentIndex.value = Math.min(
        totalCount.value,
        Math.max(1, Math.ceil((progress.value / 100) * totalCount.value))
      );
    }
  }, 900);
}

function stopProgress(done = false) {
  if (progressTimer) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
  progress.value = done ? 100 : 0;
  currentIndex.value = done ? totalCount.value : 0;
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
  payload.append("platform", form.platform);
  payload.append("style", form.style);
  return payload;
}

async function generate() {
  if (loading.value) return;
  if (!validateForm()) return;

  const payload = buildPayload();
  loading.value = true;
  results.value = [];
  emptyResultMessage.value = "";
  startProgress();

  try {
    const response = await generateDetails(payload);
    const items = Array.isArray(response.data?.items) ? response.data.items : [];
    if (!items.length) {
      emptyResultMessage.value = "后端返回 items 为空，未生成可展示结果。";
      ElMessage.warning(emptyResultMessage.value);
      return;
    }

    results.value = items;
    const failedCount = items.filter(isFailed).length;
    if (failedCount) {
      ElMessage.warning(`生成完成，${failedCount} 张图片失败，已保留失败原因`);
    } else {
      ElMessage.success("生成成功");
    }
  } catch (error) {
    console.error("[generate] 请求失败", error);
    const message = error.response?.data?.detail || error.message || "生成失败，请检查后端服务、Ollama 服务或接口配置";
    formError.value = message;
    ElMessage.error(message);
  } finally {
    stopProgress(Boolean(results.value.length));
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
    results.value = items;
    emptyResultMessage.value = items.length ? "" : "历史记录为空。";
  } catch (error) {
    console.error("[history] 请求失败", error);
    ElMessage.error("加载历史记录失败，请检查后端服务");
  }
}

function download(id) {
  window.location.href = exportUrl(id, form.platform);
}

function downloadAllZip() {
  const ids = successfulResults.value.map((item) => item.id);
  if (!ids.length) {
    ElMessage.warning("暂无可下载的成功结果");
    return;
  }
  window.location.href = exportZipUrl(ids);
}

async function copyHtml(html) {
  await navigator.clipboard.writeText(html || "");
  ElMessage.success("HTML 已复制");
}

onMounted(loadHistory);
</script>
