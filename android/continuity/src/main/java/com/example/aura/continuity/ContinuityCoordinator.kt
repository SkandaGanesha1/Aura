package com.example.aura.continuity

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ContinuityCoordinator(
    context: Context
) {
    private val discovery = DeviceDiscovery(context)

    suspend fun transfer(description: String): String = withContext(Dispatchers.IO) {
        val devices = discovery.nearbyDevices()
        if (devices.isEmpty()) {
            "No companion devices available for transfer: $description"
        } else {
            "Transferred task to ${devices.first()}: $description"
        }
    }
}
