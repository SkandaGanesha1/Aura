package com.example.aura.actuator

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Context
import android.graphics.Path
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AuraAccessibilityService : AccessibilityService() {

    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(serviceJob + Dispatchers.Main.immediate)

    override fun onServiceConnected() {
        super.onServiceConnected()
        Log.i(TAG, "Aura accessibility service connected.")
        INSTANCE = this
        serviceScope.launch {
            commandQueue.collect { command ->
                runCatching { performCommand(command) }.onFailure {
                    Log.e(TAG, "Failed to execute command: $command", it)
                }
            }
        }
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // No-op. Commands are issued programmatically by the planner.
    }

    override fun onInterrupt() {
        Log.w(TAG, "Aura accessibility service interrupted.")
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceJob.cancel()
        INSTANCE = null
    }

    private suspend fun performCommand(cmd: InputCommand) = withContext(Dispatchers.Main.immediate) {
        when (cmd) {
            is InputCommand.Click -> performClick(cmd.x, cmd.y)
            is InputCommand.Swipe -> performSwipe(cmd.startX, cmd.startY, cmd.endX, cmd.endY, cmd.durationMs)
            is InputCommand.InputText -> performTextInput(cmd.text)
        }
    }

    private fun performClick(x: Int, y: Int) {
        val path = Path().apply { moveTo(x.toFloat(), y.toFloat()) }
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, 75))
            .build()
        dispatchGesture(gesture, null, null)
    }

    private fun performSwipe(sx: Int, sy: Int, ex: Int, ey: Int, duration: Long) {
        val path = Path().apply {
            moveTo(sx.toFloat(), sy.toFloat())
            lineTo(ex.toFloat(), ey.toFloat())
        }
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, duration))
            .build()
        dispatchGesture(gesture, null, null)
    }

    private fun performTextInput(text: String) {
        // This is a placeholder: production builds should integrate with an IME or autofill service.
        Log.i(TAG, "Text input requested: $text (implement IME injection as needed).")
    }

    companion object {
        private const val TAG = "AuraAccessibilityService"
        private val commandQueue = MutableSharedFlow<InputCommand>(replay = 0, extraBufferCapacity = 16)
        @Volatile
        private var INSTANCE: AuraAccessibilityService? = null

        fun isRunning(): Boolean = INSTANCE != null

        fun ensureConfigured(context: Context) {
            // Placeholder hook for future helpers (e.g., opening accessibility settings).
            Log.d(TAG, "ensureConfigured invoked for ${context.packageName}")
        }

        fun submitCommand(command: InputCommand): Boolean = commandQueue.tryEmit(command)
    }
}
