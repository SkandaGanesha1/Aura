package com.example.aura.common

import android.app.Activity
import android.content.Intent
import android.provider.Settings

object Permissions {
    fun openAccessibilitySettings(activity: Activity) {
        val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
        activity.startActivity(intent)
    }
}
