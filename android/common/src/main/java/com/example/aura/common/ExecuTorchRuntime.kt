package com.example.aura.common

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Centralised bootstrap for loading ExecuTorch assets.
 *
 * In production this would initialise the native runtime and register model delegates.
 */
object ExecuTorchRuntime {

    private val initialised = AtomicBoolean(false)
    private lateinit var modelDirectory: File

    fun initialize(context: Context) {
        if (initialised.getAndSet(true)) {
            return
        }

        modelDirectory = File(context.filesDir, "executorch").apply { mkdirs() }
        copyAssetsIfNeeded(context)
        Log.i(TAG, "ExecuTorch runtime initialised. Assets stored at ${modelDirectory.absolutePath}")
    }

    fun getModelDirectory(): File = modelDirectory

    fun loadPlannerModule(modelFile: File): PlannerModule? {
        return modelFile.takeIf { it.exists() }?.let { PlannerModule(it) }
    }

    fun loadVisionModule(modelFile: File): VisionModule? {
        return modelFile.takeIf { it.exists() }?.let { VisionModule(it) }
    }

    private fun copyAssetsIfNeeded(context: Context) {
        val assetManager = context.assets
        val assets = assetManager.list("models") ?: return
        for (asset in assets) {
            val assetPath = "models/$asset"
            val target = File(modelDirectory, asset)
            val children = assetManager.list(assetPath)
            if (children != null && children.isNotEmpty()) {
                target.mkdirs()
                copyAssetDirectory(assetManager, assetPath, target)
            } else {
                copyAssetFile(assetManager, assetPath, target)
            }
        }
    }

    private fun copyAssetDirectory(assetManager: AssetManager, assetPath: String, targetDir: File) {
        val assets = assetManager.list(assetPath) ?: return
        for (asset in assets) {
            val childAssetPath = "$assetPath/$asset"
            val childTarget = File(targetDir, asset)
            val children = assetManager.list(childAssetPath)
            if (children != null && children.isNotEmpty()) {
                childTarget.mkdirs()
                copyAssetDirectory(assetManager, childAssetPath, childTarget)
            } else {
                copyAssetFile(assetManager, childAssetPath, childTarget)
            }
        }
    }

    private fun copyAssetFile(assetManager: AssetManager, assetPath: String, targetFile: File) {
        if (targetFile.exists()) return
        targetFile.parentFile?.mkdirs()
        try {
            assetManager.open(assetPath).use { input ->
                targetFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        } catch (exception: Exception) {
            Log.e(TAG, "Failed to copy asset $assetPath", exception)
        }
    }

    private const val TAG = "ExecuTorchRuntime"

    data class PlannerAction(val agent: String, val description: String)

    class PlannerModule(private val modelFile: File) {
        fun plan(intent: String): List<PlannerAction> {
            Log.d(TAG, "Planner ExecuTorch module loaded from ${modelFile.absolutePath}")
            val lower = intent.lowercase()
            val actions = mutableListOf<PlannerAction>()
            if ("uber" in lower || "ride" in lower) {
                actions += PlannerAction("Actuator", "Launch Uber and request a ride.")
            }
            if ("slack" in lower || "message" in lower) {
                actions += PlannerAction("Actuator", "Open Slack and notify the channel.")
            }
            if ("analyze" in lower || "see" in lower) {
                actions += PlannerAction("Perception", "Inspect the UI for relevant elements.")
            }
            if (actions.isEmpty()) {
                actions += PlannerAction("Perception", "Gather context for intent: $intent")
            }
            return actions
        }
    }

    class VisionModule(private val modelFile: File) {
        fun answer(question: String, screenshot: Bitmap?): String {
            Log.d(TAG, "Vision ExecuTorch module loaded from ${modelFile.absolutePath}")
            val resolution = if (screenshot != null) "${screenshot.width}x${screenshot.height}" else "no image"
            return "VLM answer placeholder for '$question' (screenshot: $resolution)"
        }
    }
}
