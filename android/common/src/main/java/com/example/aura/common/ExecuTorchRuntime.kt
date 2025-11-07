package com.example.aura.common

import android.content.Context
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
        val assets = context.assets.list("models") ?: return
        for (asset in assets) {
            context.assets.open("models/$asset").use { input ->
                val target = File(modelDirectory, asset)
                if (target.exists()) continue
                target.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
    }

    private const val TAG = "ExecuTorchRuntime"
}
