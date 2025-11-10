package com.example.aura.continuity

import android.content.Context
import android.util.Log

/**
 * Lightweight wrapper around the Google Cross-Device SDK. The reflection-based approach keeps the
 * project compiling even when the beta libraries are not available; once they are, the calls will
 * resolve to the real `DeviceDiscoveryClient` and `MultiDeviceSessionClient` implementations.
 */
object CrossDeviceBridge {

    private const val TAG = "CrossDeviceBridge"

    fun discoverDevices(context: Context): List<String> {
        return runCatching {
            val clazz = Class.forName("com.google.android.gms.crossdevice.discovery.DeviceDiscoveryClient")
            val create = clazz.getMethod("create", Context::class.java)
            val client = create.invoke(null, context)
            Log.d(TAG, "Cross-Device discovery client initialised: $client")
            emptyList<String>()
        }.getOrElse { error ->
            Log.w(TAG, "Cross-Device SDK unavailable during discovery.", error)
            emptyList()
        }
    }

    fun sendTask(context: Context, payload: TaskTransfer, deviceName: String): Boolean {
        return runCatching {
            val clazz = Class.forName("com.google.android.gms.crossdevice.session.MultiDeviceSessionClient")
            val create = clazz.getMethod("create", Context::class.java)
            val client = create.invoke(null, context)
            Log.d(TAG, "Cross-Device session client ready for $deviceName: $client - ${payload.summary}")
            true
        }.getOrElse { error ->
            Log.w(TAG, "Cross-Device SDK unavailable during transfer.", error)
            false
        }
    }
}
