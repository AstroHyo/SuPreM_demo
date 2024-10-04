'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Checkbox } from '@/components/ui/checkbox';
import { upload } from '@vercel/blob/client';

const targets = [
  'all',
  'spleen',
  'kidney_right',
  'kidney_left',
  'gall_bladder',
  'liver',
  'stomach',
  'aorta',
  'postcava',
  'pancreas',
];

export default function CTImageProcessor() {
  const [file, setFile] = useState<File | null>(null);
  const [selectedTargets, setSelectedTargets] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [downloadLinks, setDownloadLinks] = useState<Record<string, string>>(
    {}
  );

  const [params, setParams] = useState({
    space_x: 1.5,
    space_y: 1.5,
    space_z: 1.5,
    a_min: -175.0,
    a_max: 250.0,
    b_min: 0.0,
    b_max: 1.0,
    roi_x: 96,
    roi_y: 96,
    roi_z: 96,
    num_samples: 1,
  });

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isProcessing) {
      interval = setInterval(() => {
        setProgress((prevProgress) => {
          if (prevProgress >= 100) {
            clearInterval(interval);
            return 100;
          }
          return prevProgress + 100 / (50 * 100); // Increase by 1/300th every 100ms
        });
      }, 10);
    }
    return () => clearInterval(interval);
  }, [isProcessing]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.endsWith('.nii.gz')) {
        setFile(selectedFile);
      } else {
        alert('Please select a .nii.gz file');
      }
    }
  };

  const handleParamChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setParams((prev) => ({ ...prev, [name]: parseFloat(value) }));
  };

  const handleTargetChange = (target: string) => {
    setSelectedTargets((prev) =>
      prev.includes(target)
        ? prev.filter((t) => t !== target)
        : [...prev, target]
    );
  };

  const handleSelectAllTargets = () => {
    setSelectedTargets(targets);
  };

  const handleDeselectAllTargets = () => {
    setSelectedTargets([]);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || selectedTargets.length === 0) {
      alert('Please fill in all required fields');
      return;
    }

    setDownloadLinks({});
    setIsProcessing(true);
    setProgress(0);

    const newBlob = await upload(file.name, file, {
      access: 'public',
      handleUploadUrl: '/api/upload-ct',
      //clientPayload: JSON.stringify({ password }),
    });

    const formData = new FormData();
    formData.append('file', newBlob.url);
    //formData.append('password', password);
    formData.append('params', JSON.stringify(params));
    formData.append('selectedTargets', JSON.stringify(selectedTargets));

    console.log('formData:', formData);

    try {
      const response = await fetch('/api/process-ct', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('API request failed');
      }

      const data = await response.json();
      const links: Record<string, string> = {};

      for (const [fileName, fileContent] of Object.entries(data)) {
        const blob = new Blob([atob(fileContent as string)], {
          type: 'application/octet-stream',
        });
        const url = URL.createObjectURL(blob);
        links[fileName] = url;
      }

      setDownloadLinks(links);
    } catch (error) {
      console.error('Error processing CT image:', error);
      alert('An error occurred while processing the image');
    } finally {
      setIsProcessing(false);
      setProgress(100);
    }
  };

  return (
    <div className="min-h-screen bg-custom-bg bg-cover bg-center flex items-center justify-center">
      <div className="bg-white bg-opacity-70 p-6 rounded-lg shadow-md max-w-2xl w-full">

        <h1 className="text-3xl font-bold mb-6">TotalSegmentator Demo</h1>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <Label htmlFor="file">CT Image File (.nii.gz)</Label>
            <Input
              id="file"
              type="file"
              onChange={handleFileChange}
              accept=".nii.gz"
              disabled={isProcessing}
              required
              className="h-13"
            />
            <p className="text-sm text-muted-foreground mt-1">
              Maximum file size: 20 MB
            </p>
          </div>

          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="parameters">
              <AccordionTrigger>Optional Parameters</AccordionTrigger>
              <AccordionContent>
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(params).map(([key, value]) => (
                    <div key={key}>
                      <Label htmlFor={key}>{key}</Label>
                      <Input
                        id={key}
                        name={key}
                        type="number"
                        value={value}
                        onChange={handleParamChange}
                        disabled={isProcessing}
                      />
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <div>
            <Label>Target Selection</Label>
            <div className="flex justify-between mb-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleSelectAllTargets}
                disabled={isProcessing}
              >
                Select All
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleDeselectAllTargets}
                disabled={isProcessing}
              >
                Deselect All
              </Button>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {targets.map((target) => (
                <div key={target} className="flex items-center space-x-2">
                  <Checkbox
                    id={target}
                    checked={selectedTargets.includes(target)}
                    onCheckedChange={() => handleTargetChange(target)}
                    disabled={isProcessing}
                  />
                  <label
                    htmlFor={target}
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    {target}
                  </label>
                </div>
              ))}
            </div>
          </div>

          <Button type="submit" className="w-full" disabled={isProcessing}>
            {isProcessing ? 'Processing...' : 'Submit'}
          </Button>
        </form>

        {isProcessing && (
          <div className="mt-4">
            <Progress value={progress} className="w-full" />
            <p className="text-center mt-2">
              {progress >= 100
                ? 'Processing is taking longer than expected...'
                : `${progress.toFixed(2)}% Complete`}
            </p>
          </div>
        )}

        {Object.keys(downloadLinks).length > 0 && (
          <div className="mt-6">
            <h2 className="text-xl font-semibold mb-2">Processing Complete</h2>
            <p className="mb-4">
              Your segmentation label files are ready for download.
            </p>
            <div className="space-y-2">
              {Object.entries(downloadLinks).map(([fileName, url]) => (
                <Button key={fileName} asChild className="w-full">
                  <a href={url} download={fileName}>
                    Download {fileName}
                  </a>
                </Button>
              ))}
            </div>
          </div>
        )}
        </div>
    </div>
  );
}
