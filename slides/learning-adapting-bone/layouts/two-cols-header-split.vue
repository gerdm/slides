<script setup lang="ts">
import { computed } from 'vue'
import { useSlideContext } from '@slidev/client'

const props = defineProps<{
  class?: string
  layoutClass?: string
}>()

const { $frontmatter } = useSlideContext()

function parsePercent(value: unknown, fallback: number) {
  if (typeof value === 'number' && Number.isFinite(value))
    return `${value}%`

  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (/^\d+(?:\.\d+)?%$/.test(trimmed))
      return trimmed

    const numeric = Number(trimmed)
    if (Number.isFinite(numeric))
      return `${numeric}%`
  }

  return `${fallback}%`
}

const columnStyle = computed(() => {
  const frontmatter = ($frontmatter && 'value' in $frontmatter)
    ? ($frontmatter.value ?? {})
    : ($frontmatter ?? {})

  const left = parsePercent(frontmatter.leftCol ?? frontmatter.leftWidth, 30)
  const right = parsePercent(frontmatter.rightCol ?? frontmatter.rightWidth, 70)

  return {
    '--slidev-two-cols-header-left': left,
    '--slidev-two-cols-header-right': right,
  }
})
</script>

<template>
  <div
    class="slidev-layout two-cols-header two-cols-header-split w-full h-full"
    :class="props.layoutClass"
    :style="columnStyle"
  >
    <div class="col-header">
      <slot />
    </div>
    <div class="col-left" :class="props.class">
      <slot name="left" />
    </div>
    <div class="col-right" :class="props.class">
      <slot name="right" />
    </div>
    <div class="col-bottom" :class="props.class">
      <slot name="bottom" />
    </div>
  </div>
</template>

<style scoped>
.two-cols-header-split {
  display: grid;
  grid-template-columns: var(--slidev-two-cols-header-left, 30%) var(--slidev-two-cols-header-right, 70%);
  grid-template-rows: auto 1fr auto;
}

.col-header {
  grid-area: 1 / 1 / 2 / 3;
}

.col-left {
  grid-area: 2 / 1 / 3 / 2;
}

.col-right {
  grid-area: 2 / 2 / 3 / 3;
}

.col-bottom {
  align-self: end;
  grid-area: 3 / 1 / 3 / 3;
}
</style>