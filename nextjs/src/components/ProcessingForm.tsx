'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Checkbox } from '@/components/ui/checkbox'
import { Progress } from '@/components/ui/progress'
import { processImage } from '@/app/actions'

const targetOrgans = [
  'spleen', 'kidney_right', 'kidney_left', 'gall_bladder', 'esophagus', 
  'liver', 'stomach', 'pancreas', 'adrenal_gland_right', 'adrenal_gland_left', 
  'duodenum', 'lung_right', 'lung_left', 'colon', 'intestine', 'rectum', 
  'bladder', 'prostate', 'femur_left', 'femur_right'
]

export function ProcessingForm() {
  const [file, setFile] = useState<File | null>(null)
  const [selectedTargets, setSelectedTargets] = useState<string[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [processedFileUrl, setProcessedFileUrl] = useState<string | null>(null)

  const handleTargetChange = (target: string) => {
    setSelectedTargets(prev => 
      prev.includes(target) 
        ? prev.filter(t => t !== target)
        : [...prev, target]
    )
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file || selectedTargets.length === 0) return

    setIsProcessing(true)
    setProgress(0)
    setProcessedFileUrl(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('targets', JSON.stringify(selectedTargets))

    const progressInterval = setInterval(() => {
      setProgress(prev => Math.min(prev + 100 / 120, 100))
    }, 1000)

    try {
      const result = await processImage(formData)
      clearInterval(progressInterval)
      setProgress(100)
      setProcessedFileUrl(result.fileUrl)
    } catch (error) {
      console.error('Error processing image:', error)
      clearInterval(progressInterval)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleDownload = () => {
    if (processedFileUrl) {
      window.open(processedFileUrl, '_blank')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <Label htmlFor="file">CT/MR Data</Label>
        <Input
          id="file"
          type="file"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          accept=".dcm,.nii,.nii.gz"
        />
      </div>
      <div>
        <Label>Target Organs</Label>
        <ScrollArea className="h-[200px] w-full border rounded-md">
          <div className="p-4">
            {targetOrgans.map((organ) => (
              <div key={organ} className="flex items-center space-x-2 mb-2">
                <Checkbox
                  id={organ}
                  checked={selectedTargets.includes(organ)}
                  onCheckedChange={() => handleTargetChange(organ)}
                />
                <Label htmlFor={organ} className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                  {organ.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </Label>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>
      <Button type="submit" disabled={isProcessing || !file || selectedTargets.length === 0}>
        {isProcessing ? 'Processing...' : 'Process Image'}
      </Button>
      {isProcessing && (
        <div className="space-y-2">
          <Progress value={progress} className="w-full" />
          <p className="text-sm text-gray-500">Processing: {Math.round(progress)}%</p>
        </div>
      )}
      {processedFileUrl && (
        <Button onClick={handleDownload}>Download Processed File</Button>
      )}
    </form>
  )
}