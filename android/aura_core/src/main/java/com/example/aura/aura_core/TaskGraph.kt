package com.example.aura.aura_core

import com.squareup.moshi.JsonClass

/**
 * Immutable representation of a multi-step plan produced by the planner SLM.
 */
@JsonClass(generateAdapter = true)
data class TaskGraph(
    val intent: String,
    val steps: List<TaskStep>
) {
    fun prettyPrint(): String = buildString {
        appendLine("Intent: $intent")
        steps.forEachIndexed { index, step ->
            appendLine("${index + 1}. [${step.agent}] ${step.description}")
        }
    }
}

@JsonClass(generateAdapter = true)
data class TaskStep(
    val agent: String,
    val description: String,
    val payload: Map<String, Any?>
)
