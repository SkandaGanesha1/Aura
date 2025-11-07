package com.example.aura.aura_core

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * High-level orchestrator that owns the Sense-Plan-Act loop.
 */
class Planner(
    context: Context
) {
    private val parser = IntentParser(context)
    private val dispatcher = AgentsDispatcher(context)

    suspend fun handleIntent(intent: String): PlannerResult = withContext(Dispatchers.Default) {
        val taskGraph = parser.plan(intent)
        dispatcher.execute(taskGraph)
    }
}
