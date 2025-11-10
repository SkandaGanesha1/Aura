package com.example.aura.aura_core

import android.content.Context
import android.util.Log
import com.example.aura.common.ExecuTorchRuntime
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Wrapper responsible for invoking the planner SLM compiled with ExecuTorch.
 *
 * The current implementation uses a lightweight rule-based fallback while the
 * ExecuTorch integration is scaffolded.
 */
class IntentParser(
    private val context: Context
) {

    suspend fun plan(intent: String): TaskGraph = withContext(Dispatchers.Default) {
        loadCompiledPlanner()?.let { module ->
            runOnDevicePlanner(intent, module)
        } ?: heuristicPlan(intent)
    }

    @Volatile
    private var cachedPlanner: ExecuTorchRuntime.PlannerModule? = null

    private fun loadCompiledPlanner(): ExecuTorchRuntime.PlannerModule? {
        cachedPlanner?.let { return it }
        val directory = ExecuTorchRuntime.getModelDirectory()
        val candidate = File(directory, "planner/model.pte")
        val module = ExecuTorchRuntime.loadPlannerModule(candidate)
        cachedPlanner = module
        return module
    }

    private fun runOnDevicePlanner(intent: String, module: ExecuTorchRuntime.PlannerModule): TaskGraph {
        Log.d(TAG, "Executing planner intent via ExecuTorch module.")
        val actions = module.plan(intent)
        if (actions.isEmpty()) {
            return heuristicPlan(intent)
        }
        val steps = actions.map { action ->
            TaskStep(agent = action.agent, description = action.description, payload = emptyMap())
        }
        return TaskGraph(intent = intent, steps = steps)
    }

    private fun heuristicPlan(intent: String): TaskGraph {
        val steps = JSONArray()

        if (intent.contains("uber", ignoreCase = true)) {
            steps.put(
                JSONObject()
                    .put("agent", "Actuator")
                    .put("description", "Launch Uber and request a ride.")
            )
        }

        if (intent.contains("slack", ignoreCase = true) || intent.contains("team", ignoreCase = true)) {
            steps.put(
                JSONObject()
                    .put("agent", "Actuator")
                    .put("description", "Open Slack and notify the channel.")
            )
        }

        if (steps.length() == 0) {
            steps.put(
                JSONObject()
                    .put("agent", "Perception")
                    .put("description", "Inspect the current screen for relevant UI elements.")
            )
        }

        val mapped = (0 until steps.length()).map { index ->
            val obj = steps.getJSONObject(index)
            TaskStep(
                agent = obj.getString("agent"),
                description = obj.getString("description"),
                payload = emptyMap()
            )
        }

        return TaskGraph(
            intent = intent,
            steps = mapped
        )
    }

    companion object {
        private const val TAG = "IntentParser"
    }
}
