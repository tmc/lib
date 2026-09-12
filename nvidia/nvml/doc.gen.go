// Package nvml provides low-level Go bindings for the NVIDIA Management Library (NVML).
//
// This package provides bindings generated from nvml.h using
// [github.com/ebitengine/purego] for cgo-free dynamic loading of libnvidia-ml.
// Functions returning a C status report errors wrapping nvmlReturn_t with descriptive messages
// from nvmlErrorString.
//
// # MIG Management
//
//   - [ComputeInstance.Destroy]: Destroy compute instance.
//   - [GpuInstance.CreateComputeInstance]: Create compute instance.
//   - [GpuInstance.CreateComputeInstanceWithPlacement]: Create compute instance with the specified placement.
//   - [GpuInstance.Destroy]: Destroy GPU instance.
//   - [GpuInstance.GetComputeInstanceById]: Get compute instance for given instance ID.
//   - [GpuInstance.GetComputeInstancePossiblePlacements]: Get compute instance placements.
//   - [GpuInstance.GetComputeInstanceProfileInfo]: Get compute instance profile information.
//   - [GpuInstance.GetComputeInstanceProfileInfoV]: Versioned wrapper around [GpuInstance.GetComputeInstanceProfileInfo] that accepts a versioned [ComputeInstanceProfileInfoV2] or later output structure.
//   - [GpuInstance.GetComputeInstanceRemainingCapacity]: Get compute instance profile capacity.
//   - [GpuInstance.GetComputeInstances]: Get compute instances for given profile ID.
//   - [GpuInstance.GetInfo]: Get GPU instance information.
//
// # Device Management
//
//   - [Device.ClearAccountingPids]: Clears accounting information about all processes that have already terminated.
//   - [Device.ClearCpuAffinity]: Clear all affinity bindings for the calling thread.
//   - [Device.ClearEccErrorCounts]: Clear the ECC error and other memory error counts for the device.
//   - [Device.ClearFieldValues]: Clear values for a list of fields for a device.
//   - [Device.CreateGpuInstance]: Create GPU instance.
//   - [Device.CreateGpuInstanceWithPlacement]: Create GPU instance with the specified placement.
//   - [Device.OnSameBoard]: Check if the GPU devices are on the same physical board.
//   - [Device.RegisterEvents]: Starts recording of events on a specified devices and add the events to specified [EventSet]
//   - [Device.ResetGpuLockedClocks]: Resets the gpu clock to the default value
//   - [Device.ResetMemoryLockedClocks]: Resets the memory clock to the default value
//   - [Device.ResetNvLinkErrorCounters]: Resets all error counters to zero Please refer to [NvLinkErrorCounter] for the list of error counters that are reset
//
// # Device Queries
//
//   - [Device.GetAPIRestriction]: Retrieves the root/admin permissions on the target API.
//   - [Device.GetAccountingBufferSize]: Returns the number of processes that the circular buffer with accounting pids can hold.
//   - [Device.GetAccountingMode]: Queries the state of per process accounting mode.
//   - [Device.GetAccountingPids]: Queries list of processes that can be queried for accounting stats.
//   - [Device.GetAccountingStats]: Queries process's accounting stats (v2).
//   - [Device.GetAdaptiveClockInfoStatus]: Gets the device's Adaptive Clock status
//   - [Device.GetAddressingMode]: Get the addressing mode for a given GPU.
//   - [Device.GetArchitecture]: Get architecture for device
//   - [Device.GetAutoBoostedClocksEnabled]: Retrieve the current state of Auto Boosted clocks on a device and store it in isEnabled
//   - [Device.GetBAR1MemoryInfo]: Gets Total, Available and Used size of BAR1 memory.
//   - [Device.GetBBXTimeData]: Retrieves the cumulative number of seconds the GPU has had the driver loaded.
//   - [Device.GetBoardId]: Retrieves the device boardId from 0-N.
//   - [Device.GetBoardPartNumber]: Retrieves the the device board part number which is programmed into the board's InfoROM
//   - [Device.GetBrand]: Retrieves the brand of this device.
//   - [Device.GetBridgeChipInfo]: Get Bridge Chip Information for all the bridge chips on the board.
//   - [Device.GetBusType]: Get the type of the GPU Bus (PCIe, PCI, ...)
//   - [Device.GetC2cModeInfoV]: Retrieves the Device's C2C Mode information
//   - [Device.GetClkMonStatus]: Retrieves the frequency monitor fault status for the device.
//   - [Device.GetClock]: Retrieves the clock speed for the clock specified by the clock type and clock ID.
//   - [Device.GetClockInfo]: Retrieves the current clock speeds for the device.
//   - [Device.GetClockOffsets]: Retrieve min, max and current clock offset of some clock domain for a given PState
//   - [Device.GetComputeInstanceId]: Get compute instance ID for the given MIG device handle.
//   - [Device.GetComputeMode]: Retrieves the current compute mode for the device or MIG device.
//   - [Device.GetConfComputeGpuAttestationReport]: Get Conf Computing GPU attestation report.
//   - [Device.GetConfComputeGpuCertificate]: Get Conf Computing GPU certificate details.
//   - [Device.GetConfComputeMemSizeInfo]: Get Conf Computing Protected and Unprotected Memory Sizes.
//   - [Device.GetConfComputeProtectedMemoryUsage]: Get Conf Computing protected memory usage.
//   - [Device.GetCoolerInfo]: Retrieves the cooler's information.
//   - [Device.GetCpuAffinity]: Retrieves an array of unsigned ints (sized to cpuSetSize) of bitmasks with the ideal CPU affinity for the device For example, if processors 0, 1, 32, and 33 are ideal for the device and cpuSetSize == 2, result [0] = 0x3, result [1] = 0x3 This is equivalent to calling [Device.GetCpuAffinityWithinScope] with NVML_AFFINITY_SCOPE_NODE.
//   - [Device.GetCpuAffinityWithinScope]: Retrieves an array of unsigned ints (sized to cpuSetSize) of bitmasks with the ideal CPU affinity within node or socket for the device.
//   - [Device.GetCudaComputeCapability]: Retrieves the CUDA compute capability of the device.
//   - [Device.GetCurrPcieLinkGeneration]: Retrieves the current PCIe link generation
//   - [Device.GetCurrPcieLinkWidth]: Retrieves the current PCIe link width
//   - [Device.GetCurrentClockFreqs]: Retrieves a string with the associated current GPU Clock and Memory Clock values.
//   - [Device.GetCurrentClocksEventReasons]: Retrieves current clocks event reasons.
//   - [Device.GetDecoderUtilization]: Retrieves the current utilization and sampling size in microseconds for the Decoder
//   - [Device.GetDefaultEccMode]: Retrieves the default ECC modes for the device.
//   - [Device.GetDeviceHandleFromMigDeviceHandle]: Get parent device handle from a MIG device handle.
//   - [Device.GetDisplayActive]: Retrieves the display active state for the device.
//   - [Device.GetDisplayMode]: Retrieves the display mode for the device.
//   - [Device.GetDramEncryptionMode]: Retrieves the current and pending DRAM Encryption modes for the device.
//   - [Device.GetDynamicPstatesInfo]: Retrieve performance monitor samples from the associated subdevice.
//   - [Device.GetEccMode]: Retrieves the current and pending ECC modes for the device.
//   - [Device.GetEncoderCapacity]: Retrieves the current capacity of the device's encoder, as a percentage of maximum encoder capacity with valid values in the range 0-100.
//   - [Device.GetEncoderSessions]: Retrieves information about active encoder sessions on a target device.
//   - [Device.GetEncoderStats]: Retrieves the current encoder statistics for a given device.
//   - [Device.GetEncoderUtilization]: Retrieves the current utilization and sampling size in microseconds for the Encoder
//   - [Device.GetEnforcedPowerLimit]: Get the effective power limit that the driver enforces after taking into account all limiters
//   - [Device.GetFBCSessions]: Retrieves information about active frame buffer capture sessions on a target device.
//   - [Device.GetFBCStats]: Retrieves the active frame buffer capture sessions statistics for a given device.
//   - [Device.GetFanControlPolicy]: Gets current fan control policy.
//   - [Device.GetFanSpeedRPM]: Retrieves the intended operating speed in rotations per minute (RPM) of the device's specified fan.
//   - [Device.GetFanSpeed]: Retrieves the intended operating speed of the device's specified fan.
//   - [Device.GetFieldValues]: Request values for a list of fields for a device.
//   - [Device.GetGpcClkMinMaxVfOffset]: Retrieve the GPCCLK min max VF offset value.
//   - [Device.GetGpcClkVfOffset]: Retrieve the GPCCLK VF offset value
//   - [Device.GetGpuFabricInfoV]: Versioned wrapper around nvmlDeviceGetGpuFabricInfo that accepts a versioned [GpuFabricInfoV2] or later output structure.
//   - [Device.GetGpuInstanceById]: Get GPU instances for given instance ID.
//   - [Device.GetGpuInstanceId]: Get GPU instance ID for the given MIG device handle.
//   - [Device.GetGpuInstanceProfileInfo]: Get GPU instance profile information
//   - [Device.GetGpuInstanceProfileInfoByIdV]: GPU instance profile query function that accepts profile ID, instead of profile name.
//   - [Device.GetGpuInstanceProfileInfoV]: Versioned wrapper around [Device.GetGpuInstanceProfileInfo] that accepts a versioned [GpuInstanceProfileInfoV2] or later output structure.
//   - [Device.GetGpuInstanceRemainingCapacity]: Get GPU instance profile capacity.
//   - [Device.GetGpuInstances]: Get GPU instances for given profile ID.
//   - [Device.GetGpuMaxPcieLinkGeneration]: Retrieves the maximum PCIe link generation supported by this device
//   - [Device.GetGpuOperationMode]: Retrieves the current GOM and pending GOM (the one that GPU will switch to after reboot).
//   - [Device.GetGspFirmwareMode]: Retrieve GSP firmware mode.
//   - [Device.GetGspFirmwareVersion]: Retrieve GSP firmware version.
//   - [NewDeviceByUUID]: Acquire the handle for a particular device, based on its globally unique immutable UUID (in ASCII format) associated with each device.
//   - [NewDeviceByUUIDV]: Acquire the handle for a particular device, based on its globally unique immutable UUID (in either ASCII or binary format) associated with each device.
//   - [Device.GetHostVgpuMode]: Queries if SR-IOV host operation is supported on a vGPU supported device.
//   - [Device.GetHostname]: Get the hostname for the device.
//   - [Device.GetIndex]: Retrieves the NVML index of this device.
//   - [Device.GetInforomConfigurationChecksum]: Retrieves the checksum of the configuration stored in the device's infoROM.
//   - [Device.GetInforomImageVersion]: Retrieves the global infoROM image version
//   - [Device.GetInforomVersion]: Retrieves the version information for the device's infoROM object.
//   - [Device.GetIrqNum]: Gets the device's interrupt number
//   - [Device.GetJpgUtilization]: Retrieves the current utilization and sampling size in microseconds for the JPG
//   - [Device.GetLastBBXFlushTime]: Retrieves the timestamp and the duration of the last flush of the BBX (blackbox) infoROM object during the current run.
//   - [Device.GetMarginTemperature]: Retrieves the thermal margin temperature (distance to nearest slowdown threshold).
//   - [Device.GetMaxClockInfo]: Retrieves the maximum clock speeds for the device.
//   - [Device.GetMaxCustomerBoostClock]: Retrieves the customer defined maximum boost clock speed specified by the given clock type.
//   - [Device.GetMaxMigDeviceCount]: Get the maximum number of MIG devices that can exist under a given parent NVML device.
//   - [Device.GetMaxPcieLinkGeneration]: Retrieves the maximum PCIe link generation possible with this device and system
//   - [Device.GetMaxPcieLinkWidth]: Retrieves the maximum PCIe link width possible with this device and system
//   - [Device.GetMemClkMinMaxVfOffset]: Retrieve the MemClk (Memory Clock) min max VF offset value.
//   - [Device.GetMemClkVfOffset]: Retrieve the MemClk (Memory Clock) VF offset value.
//   - [Device.GetMemoryAffinity]: Retrieves an array of unsigned ints (sized to nodeSetSize) of bitmasks with the ideal memory affinity within node or socket for the device.
//   - [Device.GetMemoryBusWidth]: Gets the device's memory bus width
//   - [Device.GetMemoryErrorCounter]: Retrieves the requested memory error counter for the device.
//   - [Device.GetMemoryInfo]: Retrieves the amount of used, free, reserved and total memory available on the device, in bytes.
//   - [Device.GetMigDeviceHandleByIndex]: Get MIG device handle for the given index under its parent NVML device.
//   - [Device.GetMigMode]: Get MIG mode for the device.
//   - [Device.GetMinMaxClockOfPState]: Retrieve min and max clocks of some clock domain for a given PState
//   - [Device.GetMinMaxFanSpeed]: Retrieves the min and max fan speed that user can set for the GPU fan.
//   - [Device.GetMinorNumber]: Retrieves minor number for the device.
//   - [Device.GetModuleId]: Get a unique identifier for the device module on the baseboard
//   - [Device.GetMultiGpuBoard]: Retrieves whether the device is on a Multi-GPU Board Devices that are on multi-GPU boards will set multiGpuBool to a non-zero value.
//   - [Device.GetName]: Retrieves the name of this device.
//   - [Device.GetNumFans]: Retrieves the number of fans on the device.
//   - [Device.GetNumGpuCores]: Gets the device's core count
//   - [Device.GetNumaNodeId]: Get the NUMA node of the given GPU device.
//   - [Device.GetNvLinkCapability]: Retrieves the requested capability from the device's NvLink for the link specified Please refer to the [NvLinkCapability] structure for the specific caps that can be queried The return value should be treated as a boolean.
//   - [Device.GetNvLinkErrorCounter]: Retrieves the specified error counter value Please refer to [NvLinkErrorCounter] for error counters that are available
//   - [Device.GetNvLinkInfo]: Query NVLINK information associated with this device.
//   - [Device.GetNvLinkRemoteDeviceType]: Get the NVLink device type of the remote device connected over the given link.
//   - [Device.GetNvLinkState]: Retrieves the state of the device's NvLink for the link specified
//   - [Device.GetNvLinkVersion]: Retrieves the version of the device's NvLink for the link specified
//   - [Device.GetNvlinkBwMode]: Get the NvLink Reduced Bandwidth Mode for the device
//   - [Device.GetNvlinkSupportedBwModes]: Get the supported NvLink Reduced Bandwidth Modes of the device
//   - [Device.GetOfaUtilization]: Retrieves the current utilization and sampling size in microseconds for the OFA (Optical Flow Accelerator)
//   - [Device.GetP2PStatus]: Retrieve the status for a given p2p capability index between a given pair of GPU
//   - [Device.GetPciInfoExt]: Retrieves PCI attributes of this device.
//   - [Device.GetPcieLinkMaxSpeed]: Gets the device's PCIE Max Link speed in MBPS
//   - [Device.GetPcieReplayCounter]: Retrieve the PCIe replay counter.
//   - [Device.GetPcieSpeed]: Gets the device's PCIe Link speed in Mbps
//   - [Device.GetPcieThroughput]: Retrieve PCIe utilization information.
//   - [Device.GetPdi]: Retrieves the Per Device Identifier (PDI) associated with this device.
//   - [Device.GetPerformanceModes]: Retrieves a performance mode string with all the performance modes defined for this device along with their associated GPU Clock and Memory Clock values.
//   - [Device.GetPerformanceState]: Retrieves the current performance state for the device.
//   - [Device.GetPersistenceMode]: Retrieves the persistence mode associated with this device.
//   - [Device.GetPlatformInfo]: Get platform information of this device.
//   - [Device.GetPowerManagementDefaultLimit]: Retrieves default power management limit on this device, in milliwatts.
//   - [Device.GetPowerManagementLimit]: Retrieves the power management limit associated with this device.
//   - [Device.GetPowerManagementLimitConstraints]: Retrieves information about possible values of power management limits on this device.
//   - [Device.GetPowerMizerMode]: Retrieves current power mizer mode on this device.
//   - [Device.GetPowerSource]: Gets the devices power source
//   - [Device.GetPowerUsage]: Retrieves power usage for this GPU in milliwatts and its associated circuitry (e.g.
//   - [Device.GetProcessUtilization]: Retrieves the current utilization and process ID
//   - [Device.GetProcessesUtilizationInfo]: Retrieves the recent utilization and process ID for all running processes
//   - [Device.GetRemappedRows]: Get number of remapped rows.
//   - [Device.GetRepairStatus]: Get the repair status for TPC/Channel repair
//   - [Device.GetRetiredPagesPendingStatus]: Check if any pages are pending retirement and need a reboot to fully retire.
//   - [Device.GetRetiredPages]: Returns the list of retired pages by source, including pages that are pending retirement The address information provided from this API is the hardware address of the page that was retired.
//   - [Device.GetRowRemapperHistogram]: Get the row remapper histogram.
//   - [Device.GetRunningProcessDetailList]: Get information about running processes on a device for input context
//   - [Device.GetSamples]: Gets recent samples for the GPU.
//   - [Device.GetSerial]: Retrieves the globally unique board serial number associated with this device's board.
//   - [Device.GetSramEccErrorStatus]: Get SRAM ECC error status of this device.
//   - [Device.GetSupportedClocksEventReasons]: Retrieves bitmask of supported clocks event reasons that can be returned by [Device.GetCurrentClocksEventReasons]
//   - [Device.GetSupportedEventTypes]: Returns information about events supported on device
//   - [Device.GetSupportedGraphicsClocks]: Retrieves the list of possible graphics clocks that can be used as an argument for [Device.SetGpuLockedClocks].
//   - [Device.GetSupportedMemoryClocks]: Retrieves the list of possible memory clocks that can be used as an argument for [Device.SetMemoryLockedClocks].
//   - [Device.GetSupportedPerformanceStates]: Get all supported Performance States (P-States) for the device.
//   - [Device.GetTargetFanSpeed]: Retrieves the intended target speed of the device's specified fan.
//   - [Device.GetTemperatureThreshold]: Retrieves the temperature threshold for the GPU with the specified threshold type in degrees C.
//   - [Device.GetTemperatureV]: Retrieves the current temperature readings (in degrees C) for the given device.
//   - [Device.GetThermalSettings]: Used to execute a list of thermal system instructions.
//   - [Device.GetTopologyCommonAncestor]: Retrieve the common ancestor for two devices For all products.
//   - [Device.GetTopologyNearestGpus]: Retrieve the set of GPUs that are nearest to a given device at a specific interconnectivity level For all products.
//   - [Device.GetTotalEccErrors]: Retrieves the total ECC error counts for the device.
//   - [Device.GetTotalEnergyConsumption]: Retrieves total energy consumption for this GPU in millijoules (mJ) since the driver was last reloaded
//   - [Device.GetUUID]: Retrieves the globally unique immutable UUID associated with this device, as a 5 part hexadecimal string, that augments the immutable, board serial identifier.
//   - [Device.GetUnrepairableMemoryFlag]: Get the unrepairable memory flag for a given GPU
//   - [Device.GetUtilizationRates]: Retrieves the current utilization rates for the device's major subsystems.
//   - [Device.GetVbiosVersion]: Get VBIOS version of the device.
//   - [Device.GetVgpuHeterogeneousMode]: Get the vGPU heterogeneous mode for the device.
//   - [Device.GetVgpuTypeCreatablePlacements]: Query the creatable vGPU placement ID of the vGPU type.
//   - [Device.GetVgpuTypeSupportedPlacements]: Query the supported vGPU placement ID of the vGPU type.
//   - [Device.GetVirtualizationMode]: This method is used to get the virtualization mode corresponding to the GPU.
//
// # Multi Instance GPU Management
//
//   - [Device.IsMigDeviceHandle]: Test if the given handle refers to a MIG device.
//
// # Power Smoothing Information
//
//   - [Device.PowerSmoothingActivatePresetProfile]: Activiate a specific preset profile for datacenter power smoothing.
//   - [Device.PowerSmoothingSetState]: Enable or disable the Power Smoothing Feature.
//   - [Device.PowerSmoothingUpdatePresetProfileParam]: Update the value of a specific profile parameter contained within [PowerSmoothingProfile].
//
// # Device Configuration
//
//   - [Device.SetAPIRestriction]: Changes the root/admin restructions on certain APIs.
//   - [Device.SetAccountingMode]: Enables or disables per process accounting.
//   - [Device.SetAutoBoostedClocksEnabled]: Try to set the current state of Auto Boosted clocks on a device.
//   - [Device.SetClockOffsets]: Control current clock offset of some clock domain for a given PState
//   - [Device.SetComputeMode]: Set the compute mode for the device or MIG device.
//   - [Device.SetConfComputeUnprotectedMemSize]: Set Conf Computing Unprotected Memory Size.
//   - [Device.SetCpuAffinity]: Sets the ideal affinity for the calling thread and device using the guidelines given in [Device.GetCpuAffinity].
//   - [Device.SetDefaultAutoBoostedClocksEnabled]: Try to set the default state of Auto Boosted clocks on a device.
//   - [Device.SetDefaultFanSpeed]: Sets the speed of the fan control policy to default.
//   - [Device.SetDramEncryptionMode]: Set the DRAM Encryption mode for the device.
//   - [Device.SetDriverModel]: Set the driver model for the device.
//   - [Device.SetEccMode]: Set the ECC mode for the device.
//   - [Device.SetFanControlPolicy]: Sets current fan control policy.
//   - [Device.SetFanSpeed]: Sets the speed of a specified fan.
//   - [Device.SetGpuLockedClocks]: Set clocks that device will lock to.
//   - [Device.SetGpuOperationMode]: Sets new GOM.
//   - [Device.SetHostname]: Set the hostname for the device.
//   - [Device.SetMemoryLockedClocks]: Set memory clocks that device will lock to.
//   - [Device.SetMigMode]: Set MIG mode for the device.
//   - [Device.SetNvLinkDeviceLowPowerThreshold]: Set NvLink Low Power Threshold for device.
//   - [Device.SetNvlinkBwMode]: Set the NvLink Reduced Bandwidth Mode for the device
//   - [Device.SetPersistenceMode]: Set the persistence mode for the device.
//   - [Device.SetPowerManagementLimit]: Set new power limit of this device.
//   - [Device.SetPowerMizerMode]: Sets the new power mizer mode.
//   - [Device.SetTemperatureThreshold]: Sets the temperature threshold for the GPU with the specified threshold type in degrees C.
//   - [Device.SetVgpuCapabilities]: Set the desirable vGPU capability of a device
//   - [Device.SetVgpuHeterogeneousMode]: Enable or disable vGPU heterogeneous mode for the device.
//   - [Device.SetVirtualizationMode]: This method is used to set the virtualization mode corresponding to the GPU.
//
// # CPU and Memory Affinity
//
//   - [Device.ValidateInforom]: Reads the infoROM from the flash and verifies the checksums.
//   - [SystemSetConfComputeGpusReadyState]: Set Conf Computing GPUs ready state.
//   - [SystemSetConfComputeKeyRotationThresholdInfo]: Set Conf Computing key rotation threshold.
//
// # vGPU APIs
//
//   - [Device.VgpuForceGspUnload]: Executes a forced GSP unload operation on a device
//
// # Event Management
//
//   - [NewEventSet]: Create an empty set of events.
//   - [EventSet.Free]: Releases events in the set
//
// # Excluded GPU Queries
//
//   - [GetExcludedDeviceCount]: Retrieves the number of excluded GPU devices in the system.
//   - [GetExcludedDeviceInfoByIndex]: Acquire the device information for an excluded GPU device, based on its index.
//
// # GPM Management
//
//   - [GpmMetricsGet]: Calculate GPM metrics from two samples.
//   - [Device.GpmMigSampleGet]: Read a sample of GPM metrics into the provided gpmSample buffer for a MIG GPU Instance.
//   - [Device.GpmQueryDeviceSupport]: Indicate whether the supplied device supports GPM
//   - [Device.GpmQueryIfStreamingEnabled]: Get GPM stream state.
//   - [NewGpmSample]: Allocate a sample buffer to be used with NVML GPM .
//   - [GpmSample.Free]: Free an allocated sample buffer that was allocated with [NewGpmSample]
//   - [GpmSample.Get]: Read a sample of GPM metrics into the provided gpmSample buffer.
//   - [Device.GpmSetStreamingEnabled]: Set GPM stream state.
//
// # Initialization
//
//   - [InitWithFlags]: [InitWithFlags] is a variant of [Init], that allows passing a set of boolean values modifying the behaviour of [Init].
//   - [Shutdown]: Shut down NVML by releasing all GPU resources previously allocated with nvmlInit_v2().
//
// # Event Handling Methods
//
//   - [SystemEventSetCreate]: Create an empty set of system events.
//   - [SystemEventSetFree]: Releases system event set
//   - [SystemEventSetWait]: Waits on system events and delivers events
//   - [SystemRegisterEvents]: Starts recording of events on system and add the events to specified [SystemEventSet]
//
// # System Queries
//
//   - [SystemGetCPER]: Retrieves Common Platform Error Record (CPER) data.
//   - [SystemGetConfComputeCapabilities]: Get Conf Computing System capabilities.
//   - [SystemGetConfComputeGpusReadyState]: Get Conf Computing GPUs ready state.
//   - [SystemGetConfComputeKeyRotationThresholdInfo]: Get Conf Computing key rotation threshold detail.
//   - [SystemGetConfComputeSettings]: Get Conf Computing System Settings.
//   - [SystemGetConfComputeState]: Get Conf Computing System State.
//   - [SystemGetCudaDriverVersion]: Retrieves the version of the CUDA driver from the shared library.
//   - [SystemGetDriverBranch]: Retrieves the driver branch of the NVIDIA driver installed on the system.
//   - [SystemGetDriverVersion]: Retrieves the version of the system's graphics driver.
//   - [SystemGetHicVersion]: Retrieves the IDs and firmware versions for any Host Interface Cards (HICs) in the system.
//   - [SystemGetNVMLVersion]: Retrieves the version of the NVML library.
//   - [SystemGetNvlinkBwMode]: Get the global nvlink bandwith mode
//   - [SystemGetProcessName]: Gets name of the process with provided process id
//   - [SystemGetTopologyGpuSet]: Retrieve the set of GPUs that have a CPU affinity with the given CPU number For all products.
//
// # NvLink Methods
//
//   - [SystemSetNvlinkBwMode]: Set the global nvlink bandwith mode
//
// # Unit Management
//
//   - [UnitGetCount]: Retrieves the number of units in the system.
//   - [Unit.GetDevices]: Retrieves the set of GPU devices that are attached to the specified unit.
//   - [Unit.GetFanSpeedInfo]: Retrieves the fan speed readings for the unit.
//   - [NewUnitByIndex]: Acquire the handle for a particular unit, based on its index.
//   - [Unit.GetLedState]: Retrieves the LED state associated with this unit.
//   - [Unit.GetPsuInfo]: Retrieves the PSU stats for the unit.
//   - [Unit.GetTemperature]: Retrieves the temperature readings for the unit, in degrees C.
//   - [Unit.GetUnitInfo]: Retrieves the static information associated with a unit.
//   - [Unit.SetLedState]: Set the LED state for the unit.
//
// # vGPU Management
//
//   - [VgpuInstance.GetPlacementId]: Query the placement ID of active vGPU instance.
//   - [VgpuInstance.GetRuntimeStateSize]: Retrieve the currently used runtime state size of the vGPU instance
//   - [VgpuTypeId.GetFbReservation]: Retrieve the static framebuffer reservation of the vGPU type in bytes
//   - [VgpuTypeId.GetGspHeapSize]: Retrieve the static GSP heap size of the vGPU type in bytes
//
// [NVML API Reference]: https://docs.nvidia.com/deploy/nvml-api/
package nvml

// Code generated by gen-nvidia from nvml.h. DO NOT EDIT.
