package com.example.aura.perception

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import java.util.concurrent.atomic.AtomicReference
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
        val root = rootInActiveWindow ?: return
        updateRootSnapshot(root)
    }

    override fun onInterrupt() {
        _latestEvent.tryEmit("Perception service interrupted.")
    }

    companion object {
        private val _latestEvent = MutableStateFlow("Perception idle")
        val latestEvent = _latestEvent.asStateFlow()
        private val latestRoot = AtomicReference<AccessibilityNodeInfo?>()

        fun snapshotRoot(): AccessibilityNodeInfo? {
            val current = latestRoot.get() ?: return null
            return AccessibilityNodeInfo.obtain(current)
        }

        private fun updateRootSnapshot(node: AccessibilityNodeInfo) {
            val newNode = AccessibilityNodeInfo.obtain(node)
            latestRoot.getAndSet(newNode)?.recycle()
        }
    }
}
