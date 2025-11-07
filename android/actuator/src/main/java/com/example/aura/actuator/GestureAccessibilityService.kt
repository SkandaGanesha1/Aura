package com.example.aura.actuator

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.view.accessibility.AccessibilityEvent
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

class GestureAccessibilityService : AccessibilityService() {

    override fun onServiceConnected() {
        super.onServiceConnected()
        _status.tryEmit("Actuator service connected.")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // No-op. Commands are triggered programmatically.
    }

    override fun onInterrupt() {
        _status.tryEmit("Actuator service interrupted.")
    }

    fun perform(command: InputCommand) {
        when (command) {
            is InputCommand.Click -> click(command.x.toFloat(), command.y.toFloat())
            is InputCommand.Swipe -> swipe(
                command.startX.toFloat(),
                command.startY.toFloat(),
                command.endX.toFloat(),
                command.endY.toFloat(),
                command.durationMs
            )
            is InputCommand.InputText -> performGlobalAction(GLOBAL_ACTION_BACK) // Placeholder
        }
    }

    private fun click(x: Float, y: Float) {
        val path = Path().apply { moveTo(x, y) }
        dispatchGesture(GestureDescription.Builder().addStroke(GestureDescription.StrokeDescription(path, 0, 100)).build(), null, null)
    }

    private fun swipe(startX: Float, startY: Float, endX: Float, endY: Float, duration: Long) {
        val path = Path().apply {
            moveTo(startX, startY)
            lineTo(endX, endY)
        }
        dispatchGesture(GestureDescription.Builder().addStroke(GestureDescription.StrokeDescription(path, 0, duration)).build(), null, null)
    }

    companion object {
        private val _status = MutableStateFlow("Actuator idle")
        val status = _status.asStateFlow()
    }
}
