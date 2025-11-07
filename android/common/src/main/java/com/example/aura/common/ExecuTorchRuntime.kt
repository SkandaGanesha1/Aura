package com.example.aura.common

import android.content.Context
import android.content.res.AssetManager
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
}
