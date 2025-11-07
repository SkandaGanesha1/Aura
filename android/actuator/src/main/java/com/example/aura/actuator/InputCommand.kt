package com.example.aura.actuator

sealed class InputCommand {
    data class Click(val x: Int, val y: Int) : InputCommand()
    data class Swipe(val startX: Int, val startY: Int, val endX: Int, val endY: Int, val durationMs: Long) : InputCommand()
    data class InputText(val text: String) : InputCommand()
}
