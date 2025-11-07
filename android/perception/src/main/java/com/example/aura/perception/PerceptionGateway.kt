package com.example.aura.perception

import android.content.Context
import android.graphics.Bitmap
import android.view.accessibility.AccessibilityManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class PerceptionGateway(
    private val context: Context
) {
    private val accessibilityManager = context.getSystemService(Context.ACCESSIBILITY_SERVICE) as AccessibilityManager
    private val visionFallback = VisionFallback(context)

    suspend fun describeScreen(intent: String): String = withContext(Dispatchers.Default) {
        val enabledServices = accessibilityManager.getEnabledAccessibilityServiceList(AccessibilityManager.FEEDBACK_ALL_MASK)
        if (enabledServices.isEmpty()) {
            return@withContext "Accessibility service is not enabled. Cannot perceive UI for intent: $intent"
        }

        // In a full implementation we would access the current window's root node.
        val elements = AccessibilityParser.describe(null)
        if (elements.isEmpty()) {
            "Aura perception ready (no nodes parsed yet) for intent: $intent"
        } else {
            elements.joinToString(prefix = "Top UI elements:\n", separator = "\n") { element ->
                "- ${element.text.ifBlank { element.contentDescription }} @ ${element.bounds}"
            }
        }
    }

    suspend fun answerWithVision(bitmap: Bitmap, question: String): String {
        return visionFallback.answerQuestion(bitmap, question)
    }
}
