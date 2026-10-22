// Package dsv4expertcache plans bounded expert-cache slots. It does not read
// weights or allocate device memory. Backend adapters must supply real upload
// and consumer completion events; host tests do not qualify GPU event behavior.
package dsv4expertcache
