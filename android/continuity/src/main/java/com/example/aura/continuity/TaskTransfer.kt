package com.example.aura.continuity

/**
 * Payload describing the task that should hop across devices. The fields are intentionally minimal
 * so they can be serialised into the Cross-Device SDK session payload.
 */
data class TaskTransfer(
    val intent: String,
    val summary: String,
    val metadata: Map<String, String> = emptyMap()
)
