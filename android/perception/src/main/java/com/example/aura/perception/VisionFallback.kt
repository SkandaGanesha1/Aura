package com.example.aura.perception

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Placeholder for invoking the vision-language fallback model via ExecuTorch.
 */
class VisionFallback(
    private val context: Context
) {

    suspend fun answerQuestion(bitmap: Bitmap, question: String): String = withContext(Dispatchers.Default) {
        Log.d(TAG, "Running VLM fallback for question: $question (bitmap=${bitmap.width}x${bitmap.height})")
        // TODO: integrate ExecuTorch inference. For now we return a stub.
        "Vision fallback not yet implemented"
    }

    companion object {
        private const val TAG = "VisionFallback"
    }
}
