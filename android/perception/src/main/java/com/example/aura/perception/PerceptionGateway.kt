package com.example.aura.perception

import android.content.Context
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

        val rootNode = PerceptionService.snapshotRoot()
        val elements = rootNode?.let { node ->
            try {
                AccessibilityParser.describe(node)
            } finally {
                node.recycle()
            }
        } ?: emptyList()
        if (elements.isEmpty()) {
            "Unable to capture UI tree yet. Ensure Aura's perception service is enabled for intent: $intent"
        } else {
            val summary = elements.take(5).joinToString(separator = "\n") { element ->
                "- ${element.text.ifBlank { element.contentDescription.ifBlank { element.className } }} @ ${element.bounds}"
            }
            "Top UI elements:\n$summary"
        }
    }

    suspend fun answerWithVision(question: String): String = withContext(Dispatchers.Default) {
        val screenshot = ScreenshotUtils.captureWindow(context)
        visionFallback.answerQuestion(screenshot, question)
    }
}
