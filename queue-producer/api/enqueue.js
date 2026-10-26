import { send } from '@vercel/queue'

export default {
  async fetch(request) {
    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405 })
    }

    const { analysis_id: analysisId } = await request.json()
    if (typeof analysisId !== 'string' || !/^[0-9a-f-]{36}$/.test(analysisId)) {
      return new Response('Invalid analysis ID', { status: 400 })
    }

    await send('naturevision-analysis', { analysis_id: analysisId }, {
      idempotencyKey: analysisId,
    })
    return Response.json({ analysis_id: analysisId }, { status: 202 })
  },
}
