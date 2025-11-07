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

        val command = buildCommand(payload)
        Log.d(TAG, "Executing command=$command for description=$description")
        // TODO: hook into AccessibilityService performAction / GestureDescription
        "Executed: $description"
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
