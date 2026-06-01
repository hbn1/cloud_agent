<template>
  <div class="chat-container">
    <el-container class="app-shell">
      <el-aside width="260px" class="sidebar">
        <div class="sidebar-header">
          <div class="brand">
            <div class="brand-logo">CA</div>
            <h2>Cloud Agent</h2>
          </div>
          <el-button type="primary" :icon="Plus" circle @click="createNewSession" />
        </div>
        <div class="session-list">
          <div
            v-for="session in sessions"
            :key="session.id"
            :class="['session-item', { active: currentSessionId === session.id }]"
            @click="switchSession(session.id)"
          >
            <el-icon><ChatDotRound /></el-icon>
            <span class="session-name" :title="session.name">{{ session.name }}</span>
            <el-button
              class="session-close-btn"
              :icon="Close"
              circle
              size="small"
              @click.stop="deleteSession(session.id)"
            />
          </div>
        </div>
        <div class="user-info">
          <div class="mini-avatar user-avatar">U</div>
          <span class="username">user_1001</span>
        </div>
      </el-aside>

      <el-main class="chat-main">
        <div class="chat-header">
          <div class="header-title">企业云智能客服</div>
          <div class="header-subtitle">Multi-Agent · Billing · Promotion · FinOps</div>
        </div>
        <div class="message-list" ref="messageListRef">
          <div v-if="messages.length === 0" class="empty-state">
            <el-icon size="64" color="#409EFC"><Service /></el-icon>
            <h3 class="welcome-title">欢迎使用云平台智能客服</h3>
            <p class="welcome-desc">我是您的专属 AI 助手，您可以直接向我提问，或者尝试以下典型场景：</p>
            
            <div class="scenario-container">
              <el-row :gutter="20">
                <el-col :span="12">
                  <div class="scenario-card">
                    <div class="card-header">
                      <el-icon><Monitor /></el-icon>
                      <span>产品咨询与推荐</span>
                    </div>
                    <div class="scenario-list">
                      <div class="scenario-item" @click="sendQuery('云服务器ECS有哪些基本属性？')">云服务器ECS有哪些基本属性？</div>
                      <div class="scenario-item" @click="sendQuery('我是Java接口服务+MySQL，8核16G够吗？推荐具体实例型号。')">Java服务+MySQL，推荐具体实例型号</div>
                    </div>
                  </div>
                </el-col>
                <el-col :span="12">
                  <div class="scenario-card">
                    <div class="card-header">
                      <el-icon><List /></el-icon>
                      <span>账单与实例查询</span>
                    </div>
                    <div class="scenario-list">
                      <div class="scenario-item" @click="sendQuery('帮我查一下我最近的订单记录')">帮我查一下我最近的订单记录</div>
                      <div class="scenario-item" @click="sendQuery('查询我名下的所有运行中的实例')">查询我名下的所有运行中的实例</div>
                    </div>
                  </div>
                </el-col>
              </el-row>
              <el-row :gutter="20" style="margin-top: 20px;">
                <el-col :span="12">
                  <div class="scenario-card">
                    <div class="card-header">
                      <el-icon><DataLine /></el-icon>
                      <span>资源优化与降本</span>
                    </div>
                    <div class="scenario-list">
                      <div class="scenario-item" @click="sendQuery('获取近7天CPU/内存/带宽数据并做降本建议')">获取近7天资源监控并做降本建议</div>
                      <div class="scenario-item" @click="sendQuery('服务器利用率低，怎么省钱？')">服务器利用率低，怎么省钱？</div>
                    </div>
                  </div>
                </el-col>
                <el-col :span="12">
                  <div class="scenario-card">
                    <div class="card-header">
                      <el-icon><Share /></el-icon>
                      <span>产品推广活动</span>
                    </div>
                    <div class="scenario-list">
                      <div class="scenario-item" @click="sendQuery('我想推广云服务器ECS，有海报吗？')">我想推广云服务器ECS，有海报吗？</div>
                      <div class="scenario-item" @click="sendQuery('帮我生成一张 c7 计算型的推广海报')">帮我生成一张 c7 计算型的推广海报</div>
                    </div>
                  </div>
                </el-col>
              </el-row>
            </div>
          </div>

          <div 
            v-for="(msg, index) in messages" 
            :key="index"
            :class="['message-row', msg.role]"
          >
            <div :class="['msg-avatar', msg.role === 'user' ? 'user-avatar' : 'ai-avatar']">
              {{ msg.role === 'user' ? 'U' : 'AI' }}
            </div>
            <div class="message-bubble" v-html="renderMarkdown(msg.content)"></div>
          </div>
          
          <div v-if="isLoading" class="message-row assistant">
             <div class="msg-avatar ai-avatar">AI</div>
             <div class="message-bubble loading">
               <el-icon class="is-loading"><Loading /></el-icon> 正在思考与调用工具中...
             </div>
          </div>
        </div>

        <div class="input-area">
          <el-input
            v-model="inputQuery"
            type="textarea"
            :rows="3"
            placeholder="请输入您的问题，Shift + Enter 换行，Enter 发送"
            @keydown.enter.prevent="handleEnter"
            :disabled="isLoading"
          />
          <div class="input-actions">
            <el-button
              v-if="isLoading"
              type="danger"
              class="stop-btn"
              :icon="CircleClose"
              @click="stopGeneration"
            >
              停止思考
            </el-button>
            <el-button
              v-else
              type="primary"
              class="send-btn"
              :icon="Position"
              @click="sendQuery(inputQuery)"
              :disabled="!inputQuery.trim()"
            >
              发送
            </el-button>
          </div>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted } from 'vue'
import { Plus, ChatDotRound, Service, Position, Loading, Monitor, List, DataLine, Share, Close, CircleClose } from '@element-plus/icons-vue'
import { marked } from 'marked'

// 状态定义
const inputQuery = ref('')
const isLoading = ref(false)
const abortController = ref<AbortController | null>(null)
const messageListRef = ref<HTMLElement | null>(null)
const currentSessionId = ref('session_default_1')

interface Message {
  role: 'user' | 'assistant'
  content: string
}

// 按 session 存储消息，key 为 session id
const sessionMessages = ref<Record<string, Message[]>>({
  session_default_1: []
})
// 当前会话的消息（计算属性，随 currentSessionId 联动）
const messages = computed(() => sessionMessages.value[currentSessionId.value] || [])

const sessions = ref<{ id: string; name: string }[]>([
  { id: 'session_default_1', name: '新对话' }
])

const STORAGE_KEY = 'cloud_agent_sessions'

// 保存到 localStorage
const persistSessions = () => {
  const data = {
    sessions: sessions.value,
    sessionMessages: Object.fromEntries(
      Object.entries(sessionMessages.value).map(([id, msgs]) => [
        id,
        msgs.map(m => ({ role: m.role, content: m.content }))
      ])
    ),
    currentSessionId: currentSessionId.value
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
}

// 初始化：从 localStorage 恢复
onMounted(() => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      const data = JSON.parse(saved)
      if (data.sessions) sessions.value = data.sessions
      if (data.sessionMessages) sessionMessages.value = data.sessionMessages
      if (data.currentSessionId && sessions.value.some(s => s.id === data.currentSessionId)) {
        currentSessionId.value = data.currentSessionId
      }
    }
  } catch (e) {
    console.warn('Failed to restore sessions from localStorage', e)
  }
})

const createNewSession = () => {
  const newId = `session_${Date.now()}`
  sessions.value.unshift({ id: newId, name: '新对话' })
  sessionMessages.value[newId] = []
  currentSessionId.value = newId
  persistSessions()
}

const switchSession = (id: string) => {
  if (currentSessionId.value === id) return
  currentSessionId.value = id
  persistSessions()
}

const deleteSession = (id: string) => {
  const idx = sessions.value.findIndex(s => s.id === id)
  if (idx === -1) return

  sessions.value.splice(idx, 1)
  delete sessionMessages.value[id]

  // 如果删除的是当前会话，切换到最近的会话
  if (currentSessionId.value === id) {
    if (sessions.value.length > 0) {
      const targetIdx = Math.min(idx, sessions.value.length - 1)
      currentSessionId.value = sessions.value[targetIdx].id
    }
  }

  persistSessions()
}

const renderMarkdown = (text: string) => {
  return marked(text)
}

const scrollToBottom = async () => {
  await nextTick()
  if (messageListRef.value) {
    messageListRef.value.scrollTop = messageListRef.value.scrollHeight
  }
}

const handleEnter = (e: KeyboardEvent) => {
  if (e.shiftKey) return
  if (inputQuery.value.trim() && !isLoading.value) {
    sendQuery(inputQuery.value)
  }
}

const stopGeneration = () => {
  // 1. 中止前端 fetch 连接
  if (abortController.value) {
    abortController.value.abort()
    abortController.value = null
  }
  // 2. 通知后端取消（防止后台 Agent 继续运行）
  fetch('http://127.0.0.1:5000/api/chat/stop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: currentSessionId.value,
      user_id: 'user_1001'
    })
  }).catch(() => {/* ignore */})
  // 3. 移除当前不完整的助手消息
  const msgStore = sessionMessages.value[currentSessionId.value]
  if (msgStore && msgStore.length > 0) {
    const lastMsg = msgStore[msgStore.length - 1]
    if (lastMsg.role === 'assistant' && !lastMsg.content) {
      msgStore.pop()
    }
  }
  isLoading.value = false
}

const sendQuery = async (query: string) => {
  if (!query.trim()) return
  
  const text = query.trim()
  inputQuery.value = ''
  
  const msgStore = sessionMessages.value[currentSessionId.value]

  // 自动命名会话：提取第一条用户消息作为会话名
  if (msgStore?.length === 0) {
    const session = sessions.value.find(s => s.id === currentSessionId.value)
    if (session) {
      session.name = text.length > 18 ? text.slice(0, 18) + '...' : text
    }
  }

  // 添加用户消息
  msgStore.push({ role: 'user', content: text })
  scrollToBottom()
  persistSessions()

  isLoading.value = true

  // 创建 AbortController 用于取消请求
  const controller = new AbortController()
  abortController.value = controller

  // 预先创建一个空的助手消息，用于接收流式数据
  const assistantMessage: Message = { role: 'assistant', content: '' }
  msgStore.push(assistantMessage)
  const currentMsgIndex = msgStore.length - 1

  try {
    // 调用 FastAPI 后端接口并使用 fetch 接收 SSE 流
    const response = await fetch('http://127.0.0.1:5000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: text,
        user_id: 'user_1001',
        session_id: currentSessionId.value
      }),
      signal: controller.signal
    })
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body?.getReader()
    const decoder = new TextDecoder('utf-8')
    isLoading.value = false // 开始接收流，关闭 loading 状态

    if (reader) {
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // 将不完整的一行保留到下一次循环

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim()
            if (dataStr === '[DONE]') continue
            if (!dataStr) continue
            
            try {
              const data = JSON.parse(dataStr)
              if (data.content && messages.value[currentMsgIndex]) {
                messages.value[currentMsgIndex].content += data.content
                scrollToBottom()
              }
              if (data.done) {
                // 流传输完成
              }
              if (data.cancelled) {
                // 后端取消了生成，移除空消息
                if (messages.value[currentMsgIndex] && !messages.value[currentMsgIndex].content) {
                  const store = sessionMessages.value[currentSessionId.value]
                  if (store) store.pop()
                }
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e, dataStr)
            }
          }
        }
      }
    }
  } catch (error: any) {
    if (error.name === 'AbortError') {
      // 用户手动取消，不做额外处理（stopGeneration 已移除空消息）
      console.log('[Stop] 用户取消了生成')
    } else {
      console.error('API Error:', error)
      if (messages.value[currentMsgIndex]) {
        messages.value[currentMsgIndex].content = '❌ 请求失败，请检查后端服务是否启动 (FastAPI port 5000)。'
      }
    }
  } finally {
    abortController.value = null
    isLoading.value = false
    persistSessions()
    scrollToBottom()
  }
}
</script>

<style scoped>
.chat-container {
  height: 100vh;
  width: 100vw;
  background: radial-gradient(circle at 10% 20%, #e6f0ff 0%, #eef5ff 35%, #f6f8fc 100%);
  overflow: hidden;
  padding: 16px;
  box-sizing: border-box;
}
.app-shell {
  height: 100%;
  border-radius: 20px;
  overflow: hidden;
  border: 1px solid #e7ebf3;
  box-shadow: 0 20px 50px rgba(15, 35, 95, 0.08);
  background: #fff;
}
.sidebar {
  background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
  border-right: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  flex-direction: column;
}
.sidebar-header {
  padding: 18px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
}
.brand-logo {
  width: 30px;
  height: 30px;
  border-radius: 10px;
  display: grid;
  place-items: center;
  font-size: 12px;
  font-weight: 700;
  color: #fff;
  background: linear-gradient(135deg, #60a5fa, #2563eb);
}
.sidebar-header h2 {
  margin: 0;
  font-size: 16px;
  color: #f8fafc;
  letter-spacing: 0.4px;
}
.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}
.session-item {
  padding: 12px;
  margin-bottom: 8px;
  border-radius: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  color: #dbeafe;
  transition: all 0.3s;
  border: 1px solid transparent;
}
.session-item:hover {
  background-color: rgba(96, 165, 250, 0.18);
}
.session-item.active {
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.24), rgba(37, 99, 235, 0.22));
  color: #eff6ff;
  font-weight: 500;
  border-color: rgba(96, 165, 250, 0.35);
}
.user-info {
  padding: 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.12);
  display: flex;
  align-items: center;
  gap: 10px;
}
.username {
  font-weight: 600;
  color: #e2e8f0;
}

.chat-main {
  display: flex;
  flex-direction: column;
  padding: 0;
  background: linear-gradient(180deg, #f8fbff 0%, #f6f8fc 100%);
}
.chat-header {
  padding: 16px 28px 12px;
  border-bottom: 1px solid #e7edf7;
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(8px);
}
.header-title {
  font-size: 18px;
  font-weight: 700;
  color: #0f172a;
}
.header-subtitle {
  margin-top: 4px;
  color: #64748b;
  font-size: 13px;
}
.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 24px 28px;
  scroll-behavior: smooth;
}
.empty-state {
  min-height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: #64748b;
  background: #ffffff;
  border: 1px solid #e7edf7;
  border-radius: 16px;
  padding: 40px;
}
.welcome-title {
  margin-top: 16px;
  margin-bottom: 8px;
  color: #1e293b;
  font-size: 24px;
  font-weight: 600;
}
.welcome-desc {
  margin-bottom: 32px;
  color: #64748b;
  font-size: 15px;
}
.scenario-container {
  width: 100%;
  max-width: 800px;
}
.scenario-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 20px;
  height: 100%;
  transition: all 0.3s ease;
}
.scenario-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  border-color: #cbd5e1;
}
.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: #334155;
  margin-bottom: 16px;
}
.card-header .el-icon {
  color: #3b82f6;
  font-size: 20px;
}
.scenario-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.scenario-item {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 12px 16px;
  font-size: 14px;
  color: #475569;
  cursor: pointer;
  transition: all 0.2s ease;
}
.scenario-item:hover {
  background: #eff6ff;
  border-color: #93c5fd;
  color: #2563eb;
  transform: translateY(-2px);
}

.message-row {
  display: flex;
  gap: 12px;
  margin-bottom: 18px;
  max-width: 86%;
  align-items: flex-start;
}
.message-row.user {
  flex-direction: row-reverse;
  margin-left: auto;
}
.msg-avatar {
  width: 34px;
  height: 34px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  font-size: 12px;
  font-weight: 700;
  flex-shrink: 0;
}
.user-avatar {
  color: #eff6ff;
  background: linear-gradient(135deg, #3b82f6, #1d4ed8);
}
.ai-avatar {
  color: #f8fafc;
  background: linear-gradient(135deg, #0ea5e9, #22c55e);
}
.mini-avatar {
  width: 28px;
  height: 28px;
  border-radius: 9px;
  display: grid;
  place-items: center;
  font-size: 11px;
  font-weight: 700;
}
.message-bubble {
  background: #ffffff;
  padding: 13px 16px;
  border-radius: 14px;
  border: 1px solid #e7edf7;
  box-shadow: 0 8px 24px rgba(15, 35, 95, 0.05);
  line-height: 1.6;
  color: #1e293b;
  font-size: 15px;
}
.message-row.user .message-bubble {
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  color: #ffffff;
  border-color: rgba(59, 130, 246, 0.35);
}
.message-row.assistant .message-bubble {
  border-top-left-radius: 0;
}
.message-row.user .message-bubble {
  border-top-right-radius: 0;
}
.message-bubble :deep(p) { margin: 0 0 10px 0; }
.message-bubble :deep(p:last-child) { margin: 0; }
.message-bubble :deep(img) { max-width: 100%; border-radius: 8px; margin-top: 10px; }
.message-bubble :deep(pre) { background: #f4f4f5; padding: 10px; border-radius: 6px; overflow-x: auto; }
.message-bubble :deep(code) { font-family: monospace; }

.input-area {
  padding: 16px 28px 20px;
  background: #ffffff;
  border-top: 1px solid #e7edf7;
}
.input-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 12px;
}
.send-btn {
  width: 110px;
  border-radius: 10px;
}
.stop-btn {
  width: 130px;
  border-radius: 10px;
}

/* 会话关闭按钮 */
.session-close-btn {
  opacity: 0 !important;
  margin-left: auto !important;
  flex-shrink: 0 !important;
  color: #94a3b8 !important;
  --el-button-size: 22px !important;
  transition: all 0.2s !important;
}
.session-item:hover .session-close-btn {
  opacity: 1 !important;
}
.session-close-btn:hover {
  color: #ef4444 !important;
  background-color: rgba(239, 68, 68, 0.15) !important;
}
</style>
