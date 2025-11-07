package com.example.aura.perception

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

class PerceptionService : AccessibilityService() {

    override fun onServiceConnected() {
        super.onServiceConnected()
        _latestEvent.tryEmit("Perception service connected.")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event ?: return
        _latestEvent.tryEmit("Event: ${event.eventType} on ${event.className}")
    }

    override fun onInterrupt() {
        _latestEvent.tryEmit("Perception service interrupted.")
    }

    companion object {
        private val _latestEvent = MutableStateFlow("Perception idle")
        val latestEvent = _latestEvent.asStateFlow()
    }
}
