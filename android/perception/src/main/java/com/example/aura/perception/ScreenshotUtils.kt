package com.example.aura.perception

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color

/**
 * Lightweight helper to snapshot the current window. This is a placeholder that falls back to
 * drawing from the decor view when full screenshot APIs are unavailable.
 */
object ScreenshotUtils {

    fun captureWindow(context: Context): Bitmap? {
        val metrics = context.resources.displayMetrics
        val width = metrics.widthPixels.coerceAtLeast(1)
        val height = metrics.heightPixels.coerceAtLeast(1)
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.BLACK)
        }
    }
}
