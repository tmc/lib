package cudarepro

// exportRunnerSource is included in each exported package. The caller supplies
// a reviewed worker: Case does not contain enough information to infer a CUDA
// argument ABI. Output and loaded/oracle metadata are assertions from that
// worker, not independent proof of device execution or native qualification.
const exportRunnerSource = `
func readKernel(path string) ([]byte,error) {
 info,err := os.Stat(path)
 if err != nil || !info.Mode().IsRegular() { return nil,fmt.Errorf("kernel must be a regular file") }
 f,err := os.Open(path)
 if err != nil { return nil,err }
 defer f.Close()
 b,err := io.ReadAll(io.LimitReader(f,(4<<20)+1))
 if err != nil { return nil,err }
 if len(b)>4<<20 { return nil,fmt.Errorf("kernel exceeds read limit") }
 return b,nil
}

// copyRunner snapshots the reviewed executable before invocation. Native
// dynamic libraries, interpreter paths and device custody are separate gates.
func copyRunner(path,directory,want string) (string,error) {
 info,err := os.Stat(path)
 if err != nil || !info.Mode().IsRegular() { return "",fmt.Errorf("runner executable must be a regular file") }
 source,err := os.Open(path)
 if err != nil { return "",err }
 defer source.Close()
 info,err = source.Stat()
 if err != nil || !info.Mode().IsRegular() { return "",fmt.Errorf("runner executable must be a regular file") }
 destination := filepath.Join(directory,"runner")
 file,err := os.OpenFile(destination,os.O_WRONLY|os.O_CREATE|os.O_EXCL,0700)
 if err != nil { return "",err }
 hash := sha256.New()
 size,copyErr := io.Copy(io.MultiWriter(file,hash),io.LimitReader(source,(128<<20)+1))
 closeErr := file.Close()
 if copyErr != nil || closeErr != nil || size>128<<20 || fmt.Sprintf("%x",hash.Sum(nil))!=want {
  os.Remove(destination)
  return "",fmt.Errorf("refusing execution: runner executable changed or exceeds copy limit")
 }
 absolute,err := filepath.Abs(destination)
 if err != nil { os.Remove(destination);return "",err }
 return absolute,nil
}

type runnerRequest struct {
 Version int
 Attempt int
 Kernel []byte
 Case json.RawMessage
 OracleRequired bool
 OraclePatch string
}

type runnerResponse struct {
 Version int
 PID int
 KernelDigest string
 CaseDigest string
 Reference []uint64
 Observed []uint64
 Oracle struct {
  Attached bool
  PID int
  Patch string
  Evidence string
 }
}

type boundedOutput struct { buffer bytes.Buffer }
func (b *boundedOutput) Bytes() []byte { return b.buffer.Bytes() }
func (b *boundedOutput) Write(p []byte) (int,error) {
 if len(p) > (1<<20)-b.buffer.Len() { return 0,fmt.Errorf("runner output limit exceeded") }
 return b.buffer.Write(p)
}

func runCandidateAttempt(n int, kernelBytes, caseBytes []byte, policy policyData) (bool,string,error) {
 if policy.Options.RunnerPath == "" || policy.Options.RunnerSHA256 == "" {
  return false,"",fmt.Errorf("refusing execution: reviewed runner and executable SHA-256 required")
 }
 if !filepath.IsAbs(policy.Options.RunnerPath) || policy.Options.AttemptTimeout <= 0 || policy.Options.AttemptTimeout > int64(10*time.Minute) {
  return false,"",fmt.Errorf("refusing execution: invalid runner path or deadline")
 }
 if err := os.MkdirAll("receipts",0700); err != nil { return false,"",err }
 directory,err := os.MkdirTemp("receipts",fmt.Sprintf("attempt-%d-",n))
 if err != nil { return false,"",err }
 executable,err := copyRunner(policy.Options.RunnerPath,directory,policy.Options.RunnerSHA256)
 if err != nil { return false,"",err }
 defer os.Remove(executable)
 request,err := json.Marshal(runnerRequest{Version:1,Attempt:n,Kernel:kernelBytes,Case:caseBytes,OracleRequired:policy.OracleRequired,OraclePatch:policy.OraclePatch})
 if err != nil { return false,"",err }
 if len(request)>16<<20 { return false,"",fmt.Errorf("runner request limit exceeded") }
 ctx,cancel := context.WithTimeout(context.Background(),time.Duration(policy.Options.AttemptTimeout))
 defer cancel()
 cmd := exec.CommandContext(ctx,executable)
 cmd.Stdin = bytes.NewReader(request)
 var stdout,stderr boundedOutput
 cmd.Stdout,cmd.Stderr = &stdout,&stderr
 cmd.WaitDelay = time.Second
 if err := cmd.Start(); err != nil { return false,"",fmt.Errorf("start runner: %w",err) }
 pid := cmd.Process.Pid
 runErr := cmd.Wait()
 // Preserve bounded raw output before interpretation, including failed attempts.
 receipt := struct {
  Attempt int
  PID int
  RunnerSHA256 string
  KernelDigest string
  CaseDigest string
  Stdout []byte
  StdoutSHA256 string
  Stderr []byte
  Error string
  NativeQualified bool
 }{Attempt:n,PID:pid,RunnerSHA256:policy.Options.RunnerSHA256,
 KernelDigest:fmt.Sprintf("%x",sha256.Sum256(kernelBytes)),CaseDigest:fmt.Sprintf("%x",sha256.Sum256(caseBytes)),Stdout:stdout.Bytes(),StdoutSHA256:fmt.Sprintf("%x",sha256.Sum256(stdout.Bytes())),Stderr:stderr.Bytes()}
 if runErr != nil { receipt.Error = runErr.Error() }
 body,err := json.Marshal(receipt)
 if err != nil { return false,"",err }
 path := filepath.Join(directory,"receipt.json")
 if err := os.WriteFile(path,body,0600); err != nil { return false,"",err }
 if runErr != nil { return false,"",fmt.Errorf("runner failed: %w (raw receipt %s)",runErr,path) }
 var result runnerResponse
 dec := json.NewDecoder(bytes.NewReader(stdout.Bytes()))
 dec.DisallowUnknownFields()
 if err := dec.Decode(&result); err != nil { return false,"",fmt.Errorf("runner response: %w",err) }
 if err := dec.Decode(new(any)); err != io.EOF { return false,"",fmt.Errorf("runner response has trailing data") }
 if result.Version!=1 || result.PID!=pid || pid==os.Getpid() {
  return false,"",fmt.Errorf("runner response process identity mismatch")
 }
 if result.KernelDigest!=receipt.KernelDigest || result.CaseDigest!=receipt.CaseDigest {
  return false,"",fmt.Errorf("runner response artifact identity mismatch")
 }
 if policy.OracleRequired && (!result.Oracle.Attached || result.Oracle.PID!=pid || result.Oracle.Patch!=policy.OraclePatch || result.Oracle.Evidence=="") {
  return false,"",fmt.Errorf("refusing execution result: required oracle %q lacks process-bound evidence",policy.OraclePatch)
 }
 if len(result.Reference)==0 || len(result.Reference)!=len(result.Observed) {
  return false,"",fmt.Errorf("runner response missing matching output/reference arrays")
 }
 if policy.Identity.Policy!="exact-uint64" { return false,"",fmt.Errorf("unsupported comparison policy") }
 for i,want := range result.Reference {
  if result.Observed[i]!=want {
   if uint64(i)!=policy.Identity.Index { return false,fmt.Sprintf("unexpected element mismatch at index %d",i),nil }
   return true,"",nil
  }
 }
 return false,"",nil
}
`
