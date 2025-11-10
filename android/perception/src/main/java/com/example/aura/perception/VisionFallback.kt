package com.example.aura.perception

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.example.aura.common.ExecuTorchRuntime
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Minimal ExecuTorch-based VLM fallback. Uses the compiled perception model when available and falls
 * back to descriptive logs otherwise.
 */
class VisionFallback(
    private val context: Context
) {

    @Volatile
    private var cachedModule: ExecuTorchRuntime.VisionModule? = null

    suspend fun answerQuestion(bitmap: Bitmap?, question: String): String = withContext(Dispatchers.Default) {
        val module = loadVisionModule()
        module?.answer(question, bitmap) ?: run {
            val info = bitmap?.let { "${it.width}x${it.height}" } ?: "no screenshot"
            Log.w(TAG, "Vision module unavailable. Returning heuristic answer. (bitmap=$info)")
            "Vision fallback unavailable yet for question: $question"
        }
    }

    private fun loadVisionModule(): ExecuTorchRuntime.VisionModule? {
        cachedModule?.let { return it }
        val directory = ExecuTorchRuntime.getModelDirectory()
        val candidate = File(directory, "perception/vlm_fallback.pte")
        val module = ExecuTorchRuntime.loadVisionModule(candidate)
        cachedModule = module
        return module
    }

    companion object {
        private const val TAG = "VisionFallback"
    }
}
