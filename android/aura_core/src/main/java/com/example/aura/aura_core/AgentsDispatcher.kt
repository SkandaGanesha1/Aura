package com.example.aura.aura_core

import android.content.Context
import com.example.aura.actuator.GuiActuator
import com.example.aura.continuity.ContinuityCoordinator
import com.example.aura.perception.PerceptionGateway

/**
 * Dispatches planner commands to the appropriate agent implementation.
 */
class AgentsDispatcher(
    context: Context
) {
    private val perception = PerceptionGateway(context)
    private val actuator = GuiActuator(context)
    private val continuity = ContinuityCoordinator(context)

    suspend fun execute(taskGraph: TaskGraph): PlannerResult {
        val executedSteps = mutableListOf<String>()

        for (step in taskGraph.steps) {
            when (step.agent.lowercase()) {
                "perception" -> executedSteps += perception.describeScreen(step.description)
                "actuator" -> executedSteps += actuator.perform(step.description, step.payload)
                "continuity" -> executedSteps += continuity.transfer(step.description)
                else -> executedSteps += "Unsupported agent: ${step.agent}"
            }
        }

        return PlannerResult(
            intent = taskGraph.intent,
            executedSteps = executedSteps
        )
    }
}

data class PlannerResult(
    val intent: String,
    val executedSteps: List<String>
) {
    val statusMessage: String
        get() = executedSteps.joinToString(separator = "\n")
}
