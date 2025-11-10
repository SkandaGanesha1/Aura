package com.example.aura.continuity

import android.content.Context

class DeviceDiscovery(private val context: Context) {

    fun nearbyDevices(): List<String> {
        return CrossDeviceBridge.discoverDevices(context)
    }
}
