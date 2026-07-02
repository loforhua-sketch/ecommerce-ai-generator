<template>
  <main class="app-shell">
    <aside class="left-panel">
      <div class="brand">
        <div class="brand-mark">AI</div>
        <div>
          <h1>电商 AI 生成器</h1>
          <p>上传商品图片，批量生成标题、卖点、详情页文案和 HTML。</p>
        </div>
      </div>

      <el-form label-position="top" class="generator-form">
        <el-form-item label="产品图片">
          <el-upload
            v-model:file-list="fileList"
            drag
            multiple
            accept="image/*"
            :auto-upload="false"
            list-type="picture"
          >
            <el-icon class="upload-icon"><UploadFilled /></el-icon>
            <div class="el-upload__text">拖拽图片到此处，或点击选择</div>
          </el-upload>
        </el-form-item>

        <el-form-item label="商品名称">
          <el-input v-model="form.productName" placeholder="留空时由 AI 从图片识别" />
        </el-form-item>

        <div class="form-grid">
          <el-form-item label="品类">
            <el-input v-model="form.category" placeholder="如：家居百货" />
          </el-form-item>
          <el-form-item label="目标人群">
            <el-input v-model="form.audience" placeholder="如：通勤白领" />
          </el-form-item>
        </div>

        <div class="form-grid">
          <el-form-item label="售价">
            <el-input v-model="form.price" placeholder="199" />
          </el-form-item>
          <el-form-item label="原价">
            <el-input v-model="form.originPrice" placeholder="299" />
          </el-form-item>
        </div>

        <el-button type="primary" size="large" :loading="loading" @click="generate">
          <el-icon><MagicStick /></el-icon>
          批量生成
        </el-button>
      </el-form>
    </aside>

    <section class="result-panel">
      <div class="toolbar">
        <div>
          <h2>生成结果</h2>
          <span>{{ results.length ? `共 ${results.length} 条` : "等待上传图片" }}</span>
        </div>
        <el-button :icon="Refresh" @click="loadHistory">刷新历史</el-button>
      </div>

      <el-empty v-if="!results.length && !loading" description="暂无生成结果" />

      <div v-else class="result-grid">
        <article v-for="item in results" :key="item.id" class="result-card">
          <div class="image-wrap">
            <img :src="imageSrc(item)" :alt="item.product_name" />
          </div>
          <div class="card-body">
            <div class="card-head">
              <div>
                <h3>{{ item.analysis.product_name }}</h3>
                <p>{{ item.analysis.category }} · {{ item.created_at }}</p>
              </div>
              <el-tag type="success">#{{ item.id }}</el-tag>
            </div>

            <el-tabs>
              <el-tab-pane label="平台标题">
                <div class="title-list">
                  <CopyLine label="淘宝" :value="item.analysis.taobao_title" />
                  <CopyLine label="拼多多" :value="item.analysis.pdd_title" />
                  <CopyLine label="抖店" :value="item.analysis.douyin_title" />
                </div>
              </el-tab-pane>
              <el-tab-pane label="卖点文案">
                <ul class="selling-list">
                  <li v-for="point in item.analysis.selling_points" :key="point">{{ point }}</li>
                </ul>
                <p v-for="copy in item.analysis.detail_copy" :key="copy" class="copy-text">
                  {{ copy }}
                </p>
              </el-tab-pane>
              <el-tab-pane label="HTML 预览">
                <iframe :srcdoc="item.html" title="详情页预览" />
              </el-tab-pane>
            </el-tabs>

            <div class="card-actions">
              <el-button type="primary" :icon="Download" @click="download(item.id)">
                导出 HTML
              </el-button>
              <el-button :icon="DocumentCopy" @click="copyHtml(item.html)">复制 HTML</el-button>
            </div>
          </div>
        </article>
      </div>
    </section>
  </main>
</template>

<script setup>
import { onMounted, reactive, ref } from "vue";
import { DocumentCopy, Download, MagicStick, Refresh, UploadFilled } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";
import CopyLine from "./components/CopyLine.vue";
import { api, exportUrl } from "./api/client";

const loading = ref(false);
const fileList = ref([]);
const results = ref([]);
const form = reactive({
  productName: "",
  category: "",
  audience: "",
  price: "",
  originPrice: ""
});

function imageSrc(item) {
  const name = item.image_path;
  const base = import.meta.env.VITE_API_BASE_URL || "";
  return `${base}/api/uploads/${name}`;
}

async function generate() {
  if (!fileList.value.length) {
    ElMessage.warning("请先上传至少一张商品图片");
    return;
  }
  const payload = new FormData();
  fileList.value.forEach((file) => payload.append("files", file.raw));
  payload.append("product_name", form.productName);
  payload.append("category", form.category);
  payload.append("audience", form.audience);
  payload.append("price", form.price);
  payload.append("origin_price", form.originPrice);

  loading.value = true;
  try {
    const { data } = await api.post("/api/generate", payload);
    results.value = data.items;
    ElMessage.success(`已生成 ${data.items.length} 个详情页`);
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || "生成失败，请检查后端和 OpenAI API Key");
  } finally {
    loading.value = false;
  }
}

async function loadHistory() {
  const { data } = await api.get("/api/generations");
  results.value = data.items;
}

function download(id) {
  window.location.href = exportUrl(id);
}

async function copyHtml(html) {
  await navigator.clipboard.writeText(html);
  ElMessage.success("HTML 已复制");
}

onMounted(loadHistory);
</script>
