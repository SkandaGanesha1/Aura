package com.example.aura.actuator

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class GuiActuator(
    private val context: Context
) {

    suspend fun perform(description: String, payload: Map<String, Any?>): String = withContext(Dispatchers.Default) {
        if (SafetyChecker.isSensitive(description)) {
            return@withContext "Action requires confirmation: $description"
        }

        if (!AuraAccessibilityService.isRunning()) {
            AuraAccessibilityService.ensureConfigured(context)
            return@withContext "Accessibility service unavailable. Enable Aura in Settings > Accessibility to continue."
        }

        val command = buildCommand(payload)
        Log.d(TAG, "Dispatching command=$command for description=$description")
        return@withContext if (AuraAccessibilityService.submitCommand(command)) {
            "Executed: $description"
        } else {
            "Unable to execute: actuator queue is busy. Confirm Accessibility permission and retry."
        }
    }

    private fun buildCommand(payload: Map<String, Any?>): InputCommand {
        val type = payload["type"]?.toString()?.lowercase()
        return when (type) {
            "swipe" -> InputCommand.Swipe(
                startX = payload["startX"] as? Int ?: 0,
                startY = payload["startY"] as? Int ?: 0,
                endX = payload["endX"] as? Int ?: 0,
                endY = payload["endY"] as? Int ?: 0,
                durationMs = (payload["durationMs"] as? Number)?.toLong() ?: 250L
            )
            "text" -> InputCommand.InputText(payload["text"]?.toString().orEmpty())
            else -> InputCommand.Click(
                x = payload["x"] as? Int ?: 0,
                y = payload["y"] as? Int ?: 0
            )
        }
    }

    companion object {
        private const val TAG = "GuiActuator"
    }
}
