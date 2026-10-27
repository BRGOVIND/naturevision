import { QueueClient } from '@vercel/queue'

const queue = new QueueClient()

const handleMessage = queue.handleCallback(async (message) => {
  const analysisId = message.analysis_id
  if (typeof analysisId !== 'string' || !/^[0-9a-f-]{36}$/.test(analysisId)) {
    throw new Error('Invalid analysis ID in queue message')
  }

  const target = new URL(`/internal/run/${analysisId}`, process.env.NATUREVISION_BACKEND_URL)
  const response = await fetch(target, { method: 'POST' })
  if (!response.ok) {
    throw new Error(`Analysis service returned ${response.status}`)
  }
})

export default { fetch: handleMessage }
