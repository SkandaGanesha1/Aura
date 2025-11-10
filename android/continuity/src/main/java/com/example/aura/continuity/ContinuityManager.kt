package com.example.aura.continuity

import android.content.Context

class ContinuityManager(
    private val context: Context
) {

    fun transferTask(deviceName: String, payload: TaskTransfer): Boolean {
        return CrossDeviceBridge.sendTask(context, payload, deviceName)
    }
}
