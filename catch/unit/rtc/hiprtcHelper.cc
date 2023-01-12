#include "hiprtcHelper.hpp"

#include <hip_test_common.hh>

#include <iostream>
#include <fstream>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#define kernel_name "vcpy_kernel"
#define kernel_name0 "testKernel"

bool CommitBCToFile(char* executable, size_t exe_size, const std::string& bit_code_file) {
  std::fstream bc_file;
  bc_file.open(bit_code_file, std::ios::out | std::ios::binary);
  if (!bc_file) {
    std::cout << "File not created" << std::endl;
  }

  // std::cout<<"EXE SIZE: "<<exe_size<<std::endl;
  bc_file.write(executable, exe_size);

  bc_file.close();
  return true;
}

bool TestCompileProgram(const char* source_code, hiprtcProgram* prog_ptr, std::string prog_name,
                        std::string func_name) {
  hiprtcCreateProgram(prog_ptr, source_code, prog_name.c_str(), 0, nullptr, nullptr);
  hiprtcAddNameExpression(*prog_ptr, func_name.c_str());

  std::cout<<"Func Name: "<<func_name<<std::endl;

  int device = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
  std::string srdc = std::string("-fgpu-rdc");

  const char* options[] = {sarg.c_str(), srdc.c_str()};
  hiprtcResult compileResult(hiprtcCompileProgram(*prog_ptr, 2, options));

  size_t logSize;
  hiprtcGetProgramLogSize(*prog_ptr, &logSize);

  if (logSize) {
    std::string log(logSize, '\0');
    hiprtcGetProgramLog(*prog_ptr, &log[0]);
    std::cout << log << std::endl;
  }

  if (compileResult != HIPRTC_SUCCESS) {
    std::cout << "Compilation failed." << std::endl;
  }
 
  return true;
}

bool TestCompileRDC(char** bit_code_pptr, size_t* bit_code_size_ptr, const char* routine_ptr,
                    std::string routine_name) {
  hiprtcProgram prog;
  hiprtcCreateProgram(&prog, routine_ptr, routine_name.c_str(), 0, nullptr, nullptr);

  int device = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
  std::string srdc = std::string("-fgpu-rdc");

  const char* options[] = {sarg.c_str(), srdc.c_str()};

  hiprtcResult compileResult(hiprtcCompileProgram(prog, 2, options));

  size_t logSize;
  hiprtcGetProgramLogSize(prog, &logSize);

  if (logSize) {
    std::string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);
    std::cout << log << std::endl;
  }

  if (compileResult != HIPRTC_SUCCESS) {
    std::cout << "Compilation failed." << std::endl;
  }

  hiprtcGetBitcodeSize(prog, bit_code_size_ptr);
  *bit_code_pptr = new char[*bit_code_size_ptr];
  hiprtcGetBitcode(prog, *bit_code_pptr);

  return true;
}

bool TestModuleLoadData(void* cuOut) {
  bool test_passed = true;

  size_t LEN = 64;
  size_t SIZE = sizeof(float) * LEN;
  float* A = nullptr;
  float* B = nullptr;

  hipDeviceptr_t Ad, Bd;
  A = new float[LEN];
  B = new float[LEN];

  for (size_t idx = 0; idx < LEN; ++idx) {
    A[idx] = idx;
    B[idx] = 0;
  }

  for (size_t idx = 0; idx < LEN; ++idx) {
    // std::cout<<"AT_INIT --> idx: "<<idx<<" A: "<<A[idx]<<" B: "<<B[idx]<<std::endl;
  }

  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIPCHECK(hipMemcpyHtoD(Ad, A, SIZE));
  HIPCHECK(hipMemcpyHtoD(Bd, B, SIZE));

  hipStream_t hip_stream;
  HIPCHECK(hipStreamCreate(&hip_stream));

  struct {
    void* _Ad;
    void* _Bd;
  } args;
  args._Ad = reinterpret_cast<void*>(Ad);
  args._Bd = reinterpret_cast<void*>(Bd);
  size_t args_size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &args_size, HIP_LAUNCH_PARAM_END};

  hipModule_t hip_module = nullptr;
  hipFunction_t hip_function = nullptr;
  HIPCHECK(hipModuleLoadData(&hip_module, cuOut));
  HIPCHECK(hipModuleGetFunction(&hip_function, hip_module, kernel_name));
  HIPCHECK(hipModuleLaunchKernel(hip_function, 1, 1, 1, LEN, 1, 1, 0, hip_stream, NULL,
                                 reinterpret_cast<void**>(&config)));
  HIPCHECK(hipStreamSynchronize(hip_stream));

  HIPCHECK(hipMemcpyDtoH(B, Bd, SIZE));

  for (size_t idx = 0; idx < LEN; ++idx) {
    if (A[idx] != B[idx]) {
      test_passed = false;
      std::cout << "FAIL --> idx: " << idx << " A: " << A[idx] << " B: " << B[idx] << std::endl;
      break;
    } else {
      // std::cout<<"PASS --> idx: "<<idx<<" A: "<<A[idx]<<" B: "<<B[idx]<<std::endl;
    }
  }

  HIPCHECK(hipStreamDestroy(hip_stream));
  HIPCHECK(hipModuleUnload(hip_module));

  return test_passed;
}

bool TestModuleLoad2Data(void* cuOut) {
  bool test_passed = true;

  int* x = nullptr;
  HIPCHECK(hipMalloc(&x, sizeof(int)));
  HIPCHECK(hipMemset(x, 0x00, sizeof(int)));

  struct {
    void* _x;
  } args;
  args._x = reinterpret_cast<void*>(x);
  size_t args_size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &args_size, HIP_LAUNCH_PARAM_END};

  hipModule_t hip_module = nullptr;
  hipFunction_t hip_function = nullptr;
  HIPCHECK(hipModuleLoadData(&hip_module, cuOut));
  HIPCHECK(hipModuleGetFunction(&hip_function, hip_module, kernel_name0));
  HIPCHECK(hipModuleLaunchKernel(hip_function, 1, 1, 1, 1, 1, 1, 0, NULL, NULL,
                                 reinterpret_cast<void**>(&config)));
  HIPCHECK(hipDeviceSynchronize());

  HIPCHECK(hipModuleUnload(hip_module));
  HIPCHECK(hipFree(x));
  return test_passed;
}

bool OpenFileHandle(const char* fname, FileDesc* fd_ptr, size_t* sz_ptr) {
  if ((fd_ptr == nullptr) || (sz_ptr == nullptr)) {
    std::cout << "Invalid arguments, fname: " << fname << " fd_ptr: " << fd_ptr
              << "sz_ptr: " << sz_ptr << std::endl;
    return false;
  }

#if defined(_WIN32)
  *fd_ptr = INVALID_HANDLE_VALUE;
  *fd_ptr =
      CreateFileA(fname, GENERIC_READ, 0x1, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, NULL);
  if (*fd_ptr == INVALID_HANDLE_VALUE) {
    return false;
  }

  *sz_ptr = GetFileSize(*fd_ptr, NULL);

#else
  // open system function call, return false on fail
  struct stat stat_buf;
  *fd_ptr = open(fname, O_RDONLY);
  if (*fd_ptr < 0) {
    return false;
  }

  // Retrieve stat info and size
  if (fstat(*fd_ptr, &stat_buf) != 0) {
    close(*fd_ptr);
    return false;
  }

  *sz_ptr = stat_buf.st_size;
#endif

  return true;
}

bool CloseFileHandle(FileDesc fdesc) {
#if defined(_WIN32)
  // return false on failure
  if (CloseHandle(fdesc) < 0) {
    return false;
  }
#else
  // Return false if close system call fails
  if (close(fdesc) < 0) {
    return false;
  }
#endif
  return true;
}

bool MemoryMapFileDesc(FileDesc fdesc, size_t fsize, size_t foffset, const void** mmap_pptr) {
#if defined(_WIN32)
  if (fdesc == INVALID_HANDLE_VALUE) {
    return false;
  }

  HANDLE map_handle = CreateFileMappingA(fdesc, NULL, PAGE_READONLY, 0, 0, NULL);
  if (map_handle == INVALID_HANDLE_VALUE) {
    CloseHandle(map_handle);
    return false;
  }

  *mmap_pptr = MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, 0);
#else
  if (fdesc <= 0) {
    return false;
  }

  // If the offset is not aligned then align it
  // and recalculate the new size
  if (foffset > 0) {
    size_t old_foffset = foffset;
    foffset = alignUp(foffset, 4096);
    fsize += (foffset - old_foffset);
  }

  *mmap_pptr = mmap(NULL, fsize, PROT_READ, MAP_SHARED, fdesc, foffset);
#endif
  return true;
}

bool MemoryUnmapFile(const void* mmap_ptr, size_t mmap_size) {
#if defined(_WIN32)
  if (!UnmapViewOfFile(mmap_ptr)) {
    std::cout << "Unmap file failed: " << mmap_ptr << std::endl;
    return false;
  }
#else
  if (munmap(const_cast<void*>(mmap_ptr), mmap_size) != 0) {
    std::cout << "Unmap file failed: " << mmap_ptr << std::endl;
    return false;
  }
#endif
  return true;
}

bool GetMapPtr(const char* fname, FileDesc* fd_ptr, size_t* sz_ptr, const void** mmap_pptr) {
  if (!OpenFileHandle(fname, fd_ptr, sz_ptr)) {
    std::cout << "Opening File handle Failed: " << fname << std::endl;
    return false;
  }

  if (!MemoryMapFileDesc(*fd_ptr, *sz_ptr, 0, mmap_pptr)) {
    std::cout << "Memmap failed: fd_ptr: " << *fd_ptr << "fd_size" << *sz_ptr << std::endl;
    return false;
  }

  if (!CloseFileHandle(*fd_ptr)) {
    std::cout << "Closing the file handle Failed: " << *fd_ptr << std::endl;
    return false;
  }

  return true;
}

bool DelMapPtr(const void* mmap_ptr, size_t mmap_size) {
  return MemoryUnmapFile(mmap_ptr, mmap_size);
}
