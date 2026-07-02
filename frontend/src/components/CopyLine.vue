<template>
  <div class="copy-line">
    <strong>{{ label }}</strong>
    <span>{{ value }}</span>
    <el-button size="small" :icon="DocumentCopy" @click="copy">复制</el-button>
  </div>
</template>

<script setup>
import { DocumentCopy } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";

const props = defineProps({
  label: { type: String, required: true },
  value: { type: String, default: "" }
});

async function copy() {
  await navigator.clipboard.writeText(props.value);
  ElMessage.success(`${props.label}标题已复制`);
}
</script>

